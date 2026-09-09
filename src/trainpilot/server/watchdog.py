"""Server-side Watchdog service for background health checks and housekeeping."""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from trainpilot.common.states import TaskState
from trainpilot.server.config import ServerSettings, settings
from trainpilot.server.feishu.client import default_feishu_client
from trainpilot.server.mailbox import TaskMailboxManager, default_mailbox

logger = logging.getLogger("trainpilot.watchdog")


class ServerWatchdog:
    """Background daemon thread that periodically checks task health and trims database.

    Responsibilities:
    1. Scan tasks for missing heartbeats (stale / crashed / NCCL hang).
    2. Dispatch stale alert cards to Feishu (with debouncing so each stall alerts once).
    3. Perform periodic SQLite retention trimming to keep database lean.
    """

    def __init__(
        self,
        mailbox: Optional[TaskMailboxManager] = None,
        server_settings: Optional[ServerSettings] = None,
    ):
        self.mailbox = mailbox or default_mailbox
        self.settings = server_settings or settings
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._checks_count = 0
        self._stale_detected_total = 0
        self._stale_alerts_sent_total = 0
        self._last_check_at: Optional[str] = None
        self._alerted_task_ids: set = set()
        self._tasks_expired_total = 0
        self._last_cleanup_at: Optional[str] = None
        self._last_cleanup_monotonic: Optional[float] = None
        self._state_lock = threading.Lock()
        self._pending_alert_task_ids: set = set()
        self._alert_slots = threading.BoundedSemaphore(4)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_stats(self) -> Dict[str, Any]:
        """Return runtime statistics of the watchdog."""
        return {
            "running": self.is_running,
            "checks_count": self._checks_count,
            "stale_detected_total": self._stale_detected_total,
            "stale_alerts_sent_total": self._stale_alerts_sent_total,
            "last_check_at": self._last_check_at,
            "interval_seconds": self.settings.watchdog_interval_seconds,
            "heartbeat_timeout_seconds": self.settings.task_heartbeat_timeout_seconds,
            "task_cleanup_enabled": self.settings.enable_task_cleanup,
            "task_retention_hours": self.settings.task_retention_hours,
            "task_cleanup_interval_seconds": self.settings.task_cleanup_interval_seconds,
            "tasks_expired_total": self._tasks_expired_total,
            "last_cleanup_at": self._last_cleanup_at,
        }

    def start(self) -> None:
        """Start the watchdog background loop if enabled."""
        if not self.settings.enable_watchdog and not self.settings.enable_task_cleanup:
            logger.info("Watchdog and task cleanup are disabled by configuration")
            return
        if self.is_running:
            logger.warning("Watchdog is already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="TrainPilot-Watchdog")
        self._thread.start()
        logger.info(
            "Started TrainPilot maintenance service (watchdog: %s, task cleanup: %s)",
            self.settings.enable_watchdog,
            self.settings.enable_task_cleanup,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Gracefully stop the watchdog background thread."""
        if not self.is_running:
            return
        logger.info("Stopping TrainPilot Server Watchdog...")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._thread = None
        logger.info("TrainPilot Server Watchdog stopped.")

    def run_check_once(self) -> Dict[str, Any]:
        """Perform a single inspection sweep (can be called directly or from tests)."""
        now_dt = datetime.now(timezone.utc)
        self._last_check_at = now_dt.isoformat()
        self._checks_count += 1

        timeout_sec = self.settings.task_heartbeat_timeout_seconds
        stale_tasks = (
            self.mailbox.get_stale_tasks(timeout_seconds=timeout_sec)
            if self.settings.enable_watchdog
            else []
        )
        self._stale_detected_total += len(stale_tasks)

        alerts_sent_this_round = 0
        active_stale_ids = set()

        for t in stale_tasks:
            active_stale_ids.add(t.task_id)
            # Check if this task was already alerted (debounce)
            with self._state_lock:
                is_already_alerted = (
                    t.task_id in self._alerted_task_ids
                    or t.task_id in self._pending_alert_task_ids
                )
            if not is_already_alerted and self.mailbox._storage:
                is_already_alerted = self.mailbox._storage.is_task_stale_alerted(t.task_id)

            if not is_already_alerted:
                # Calculate silent duration
                last_ts = t.last_heartbeat_at or t.updated_at
                silent_seconds = timeout_sec
                if last_ts:
                    try:
                        parsed = datetime.fromisoformat(last_ts)
                        if parsed.tzinfo is None:
                            parsed = parsed.replace(tzinfo=timezone.utc)
                        silent_seconds = max((now_dt - parsed).total_seconds(), float(timeout_sec))
                    except Exception:
                        pass

                logger.warning("Watchdog detected stalled task: %s (silent for %.1fs, state: %s)",
                               t.task_id, silent_seconds, t.state.value)

                # Direct calls remain synchronous and deterministic for operators/tests.
                # The running maintenance loop uses bounded daemon workers so a slow
                # Feishu request cannot block subsequent scans and cleanup sweeps.
                if self.is_running and self._alert_slots.acquire(blocking=False):
                    with self._state_lock:
                        self._pending_alert_task_ids.add(t.task_id)
                    threading.Thread(
                        target=self._deliver_stale_alert,
                        args=(t, silent_seconds, True),
                        daemon=True,
                        name=f"TrainPilot-StaleAlert-{t.task_id[:24]}",
                    ).start()
                elif not self.is_running:
                    if self._deliver_stale_alert(t, silent_seconds):
                        alerts_sent_this_round += 1

        # Clear alert debounce for tasks that are no longer stale (e.g., resumed heartbeats)
        with self._state_lock:
            recovered_ids = self._alerted_task_ids - active_stale_ids
        for r_id in recovered_ids:
            with self._state_lock:
                self._alerted_task_ids.discard(r_id)
            if self.mailbox._storage:
                self.mailbox._storage.set_task_stale_alerted(r_id, False)

        # Retention trimming on SQLite database
        trimmed = 0
        cleaned_tasks = 0
        vacuum_reclaimed = False
        if self.mailbox._storage:
            cleanup_due = (
                self.settings.enable_task_cleanup
                and (
                    self._last_cleanup_monotonic is None
                    or time.monotonic() - self._last_cleanup_monotonic
                    >= self.settings.task_cleanup_interval_seconds
                )
            )
            if cleanup_due:
                try:
                    # append_event already enforces the cap. This periodic pass
                    # mainly handles a reduced cap after configuration changes,
                    # so a full GROUP BY scan is unnecessary every watchdog tick.
                    trimmed = self.mailbox._storage.trim_all_events(
                        max_events=self.settings.max_events_per_task
                    )
                    retention_seconds = self.settings.task_retention_hours * 3600
                    expired_ids = self.mailbox.clean_expired_tasks(retention_seconds)
                    cleaned_tasks = len(expired_ids)
                    self._tasks_expired_total += cleaned_tasks
                    with self._state_lock:
                        self._alerted_task_ids.difference_update(expired_ids)
                    self._last_cleanup_at = datetime.now(timezone.utc).isoformat()
                    self._last_cleanup_monotonic = time.monotonic()
                except Exception as exc:
                    logger.error("Failed to clean expired tasks: %s", exc)

            # Reclaim pages after deletes, not before them.
            if trimmed > 0 or cleaned_tasks > 0:
                try:
                    self.mailbox._storage.checkpoint_and_vacuum()
                    vacuum_reclaimed = True
                except Exception as exc:
                    logger.debug("Storage checkpoint/vacuum maintenance: %s", exc)

        # Evict inactive terminal tasks from memory to prevent RAM accumulation
        if hasattr(self.mailbox, "_evict_cold_tasks_locked"):
            with self.mailbox._lock:
                self.mailbox._evict_cold_tasks_locked()

        return {
            "stale_count": len(stale_tasks),
            "alerts_sent": alerts_sent_this_round,
            "events_trimmed": trimmed,
            "tasks_expired": cleaned_tasks,
            "vacuum_reclaimed": vacuum_reclaimed,
            "checked_at": self._last_check_at,
        }

    def _deliver_stale_alert(
        self, task: Any, silent_seconds: float, background: bool = False
    ) -> bool:
        """Deliver one stale alert and update debounce state after success."""
        try:
            default_feishu_client.send_stale_alert(
                task_id=task.task_id,
                silent_seconds=silent_seconds,
                last_heartbeat_at=task.last_heartbeat_at,
                latest_step=task.latest_step,
                latest_epoch=task.latest_epoch,
                latest_message=task.latest_message,
            )
            if self.mailbox._storage:
                self.mailbox._storage.set_task_stale_alerted(task.task_id, True)
            with self._state_lock:
                self._alerted_task_ids.add(task.task_id)
                self._stale_alerts_sent_total += 1
            return True
        except Exception as exc:
            logger.error("Failed to send stale alert for task %s: %s", task.task_id, exc)
            return False
        finally:
            if background:
                with self._state_lock:
                    self._pending_alert_task_ids.discard(task.task_id)
                self._alert_slots.release()

    def _run_loop(self) -> None:
        """Main periodic loop."""
        enabled_intervals = []
        if self.settings.enable_watchdog:
            enabled_intervals.append(float(self.settings.watchdog_interval_seconds))
        if self.settings.enable_task_cleanup:
            enabled_intervals.append(float(self.settings.task_cleanup_interval_seconds))
        interval = max(1.0, min(enabled_intervals))
        while not self._stop_event.is_set():
            try:
                self.run_check_once()
            except Exception as exc:
                logger.error("Error during watchdog check sweep: %s", exc)
            self._stop_event.wait(timeout=interval)


# Global singleton watchdog instance
default_watchdog = ServerWatchdog()
