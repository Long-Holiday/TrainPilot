"""Small bounded daemon-worker pool for non-critical external notifications."""

import logging
import queue
import threading
import time
from typing import Any, Callable

logger = logging.getLogger("trainpilot.background")



class BackgroundDispatcher:
    """Run best-effort jobs without delaying API responses or growing unbounded."""

    def __init__(self, workers: int = 4, capacity: int = 1000):
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, capacity))
        self._lock = threading.Lock()
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._dropped = 0
        for index in range(max(1, workers)):
            threading.Thread(
                target=self._run,
                daemon=True,
                name=f"TrainPilot-Background-{index + 1}",
            ).start()

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
        """Queue a job, returning false instead of blocking when saturated."""
        try:
            self._queue.put_nowait((fn, args, kwargs))
            with self._lock:
                self._submitted += 1
            return True
        except queue.Full:
            with self._lock:
                self._dropped += 1
            logger.error("Background queue is full; dropping job %s", fn.__name__)
            return False

    def get_stats(self) -> dict[str, int]:
        """Expose queue pressure and delivery outcomes for health monitoring."""
        with self._lock:
            return {
                "queued": self._queue.qsize(),
                "submitted_total": self._submitted,
                "completed_total": self._completed,
                "failed_total": self._failed,
                "dropped_total": self._dropped,
            }

    def wait_idle(self, timeout: float = 2.0) -> bool:
        """Wait until all queued background jobs have completed execution."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._queue.unfinished_tasks == 0:
                    return True
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    def _run(self) -> None:
        while True:
            fn, args, kwargs = self._queue.get()
            try:
                fn(*args, **kwargs)
            except Exception:
                with self._lock:
                    self._failed += 1
                logger.exception("Background job %s failed", fn.__name__)
            else:
                with self._lock:
                    self._completed += 1
            finally:
                self._queue.task_done()


feishu_dispatcher = BackgroundDispatcher()
