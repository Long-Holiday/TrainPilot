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
        }

    def start(self) -> None:
        """Start the watchdog background loop if enabled."""
        if not self.settings.enable_watchdog:
            logger.info("Watchdog is disabled by configuration (enable_watchdog=False)")
            return
        if self.is_running:
            logger.warning("Watchdog is already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="TrainPilot-Watchdog")
        self._thread.start()
        logger.info("Started TrainPilot Server Watchdog (interval: %ss, heartbeat_timeout: %ss)",
                    self.settings.watchdog_interval_seconds, self.settings.task_heartbeat_timeout_seconds)

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
        stale_tasks = self.mailbox.get_stale_tasks(timeout_seconds=timeout_sec)
        self._stale_detected_total = len(stale_tasks)

        alerts_sent_this_round = 0
        active_stale_ids = set()

        for t in stale_tasks:
            active_stale_ids.add(t.task_id)
            # Check if this task was already alerted (debounce)
            is_already_alerted = t.task_id in self._alerted_task_ids
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

                # Send stale alert card
                try:
                    default_feishu_client.send_stale_alert(
                        task_id=t.task_id,
                        silent_seconds=silent_seconds,
                        last_heartbeat_at=t.last_heartbeat_at,
                        latest_step=t.latest_step,
                        latest_epoch=t.latest_epoch,
                        latest_message=t.latest_message,
                    )
                    self._alerted_task_ids.add(t.task_id)
                    if self.mailbox._storage:
                        self.mailbox._storage.set_task_stale_alerted(t.task_id, True)
                    self._stale_alerts_sent_total += 1
                    alerts_sent_this_round += 1
                except Exception as exc:
                    logger.error("Failed to send stale alert for task %s: %s", t.task_id, exc)

        # Clear alert debounce for tasks that are no longer stale (e.g., resumed heartbeats)
        recovered_ids = self._alerted_task_ids - active_stale_ids
        for r_id in recovered_ids:
            self._alerted_task_ids.discard(r_id)
            if self.mailbox._storage:
                self.mailbox._storage.set_task_stale_alerted(r_id, False)

        # Retention trimming on SQLite database
        trimmed = 0
        cleaned_tasks = 0
        vacuum_reclaimed = False
        if self.mailbox._storage:
            try:
                trimmed = self.mailbox._storage.trim_all_events(max_events=self.settings.max_events_per_task)
            except Exception as exc:
                logger.error("Failed to trim excess events during watchdog run: %s", exc)

            # Periodic disk and WAL space reclamation (every 10 sweeps or when events were trimmed)
            if self._checks_count % 10 == 0 or trimmed > 0:
                try:
                    if hasattr(self.mailbox._storage, "checkpoint_and_vacuum"):
                        self.mailbox._storage.checkpoint_and_vacuum()
                        vacuum_reclaimed = True
                    if hasattr(self.mailbox._storage, "clean_expired_tasks"):
                        cleaned_tasks = self.mailbox._storage.clean_expired_tasks()
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

    def _run_loop(self) -> None:
        """Main periodic loop."""
        interval = max(1.0, float(self.settings.watchdog_interval_seconds))
        while not self._stop_event.is_set():
            try:
                self.run_check_once()
            except Exception as exc:
                logger.error("Error during watchdog check sweep: %s", exc)
            self._stop_event.wait(timeout=interval)


# Global singleton watchdog instance
default_watchdog = ServerWatchdog()
