"""Background cleanup service: periodic task and database retention housekeeping."""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from trainpilot.server.config import ServerSettings, settings
from trainpilot.server.mailbox import TaskMailboxManager, default_mailbox

logger = logging.getLogger("trainpilot.cleanup")


class TaskCleanupService:
    """Background daemon that cleans expired tasks and trims the database."""

    def __init__(
        self,
        mailbox: Optional[TaskMailboxManager] = None,
        server_settings: Optional[ServerSettings] = None,
    ):
        self.mailbox = mailbox or default_mailbox
        self.settings = server_settings or settings
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tasks_expired_total = 0
        self._plans_expired_total = 0
        self._last_cleanup_at: Optional[str] = None
        self._last_cleanup_monotonic: Optional[float] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_stats(self) -> Dict[str, Any]:
        """Return runtime statistics of the cleanup service."""
        return {
            "running": self.is_running,
            "task_cleanup_enabled": self.settings.enable_task_cleanup,
            "task_retention_hours": self.settings.task_retention_hours,
            "task_cleanup_interval_seconds": self.settings.task_cleanup_interval_seconds,
            "plan_retention_hours": self.settings.plan_retention_hours,
            "tasks_expired_total": self._tasks_expired_total,
            "plans_expired_total": self._plans_expired_total,
            "last_cleanup_at": self._last_cleanup_at,
        }

    def start(self) -> None:
        """Start the cleanup background loop if enabled."""
        if not self.settings.enable_task_cleanup:
            logger.info("Task cleanup is disabled by configuration")
            return
        if self.is_running:
            logger.warning("Task cleanup service is already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="TrainPilot-TaskCleanup"
        )
        self._thread.start()
        logger.info(
            "Started TrainPilot task cleanup service (retention: %sh, interval: %ss)",
            self.settings.task_retention_hours,
            self.settings.task_cleanup_interval_seconds,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Gracefully stop the cleanup background thread."""
        if not self.is_running:
            return
        logger.info("Stopping TrainPilot task cleanup service...")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._thread = None
        logger.info("TrainPilot task cleanup service stopped.")

    def run_cleanup_once(self) -> Dict[str, Any]:
        """Perform a single maintenance and cleanup sweep."""
        trimmed = 0
        cleaned_tasks = 0
        cleaned_plans = 0
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
                    trimmed = self.mailbox._storage.trim_all_events(
                        max_events=self.settings.max_events_per_task
                    )
                    retention_seconds = self.settings.task_retention_hours * 3600
                    expired_ids = self.mailbox.clean_expired_tasks(retention_seconds)
                    cleaned_tasks = len(expired_ids)
                    self._tasks_expired_total += cleaned_tasks
                    # Acknowledged next-task plans only; pending ones never expire.
                    cleaned_plans = self.mailbox._storage.delete_expired_plans(
                        max_age_seconds=self.settings.plan_retention_hours * 3600
                    )
                    self._plans_expired_total += cleaned_plans
                    self._last_cleanup_at = datetime.now(timezone.utc).isoformat()
                    self._last_cleanup_monotonic = time.monotonic()
                except Exception as exc:
                    logger.error("Failed to clean expired tasks: %s", exc)

            if trimmed > 0 or cleaned_tasks > 0 or cleaned_plans > 0:
                try:
                    self.mailbox._storage.checkpoint_and_vacuum()
                    vacuum_reclaimed = True
                except Exception as exc:
                    logger.debug("Storage checkpoint/vacuum maintenance: %s", exc)

        if hasattr(self.mailbox, "_evict_cold_tasks_locked"):
            with self.mailbox._lock:
                self.mailbox._evict_cold_tasks_locked()

        return {
            "events_trimmed": trimmed,
            "tasks_expired": cleaned_tasks,
            "plans_expired": cleaned_plans,
            "vacuum_reclaimed": vacuum_reclaimed,
            "last_cleanup_at": self._last_cleanup_at,
        }

    def _run_loop(self) -> None:
        """Main periodic loop."""
        interval = max(1.0, float(self.settings.task_cleanup_interval_seconds))
        while not self._stop_event.is_set():
            try:
                self.run_cleanup_once()
            except Exception as exc:
                logger.error("Error during task cleanup sweep: %s", exc)
            self._stop_event.wait(timeout=interval)


default_cleanup = TaskCleanupService()
