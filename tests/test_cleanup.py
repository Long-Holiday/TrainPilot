"""Unit tests for TaskCleanupService and retention housekeeping."""

import time
import pytest

from trainpilot.common.states import TaskState
from trainpilot.server.cleanup import TaskCleanupService
from trainpilot.server.config import ServerSettings
from trainpilot.server.mailbox import TaskMailboxManager, default_mailbox
from trainpilot.server.storage import SQLiteStorage


@pytest.fixture(autouse=True)
def clean_env():
    default_mailbox.reset()
    yield
    default_mailbox.reset()


def test_scheduled_task_cleanup_runs(tmp_path):
    """Cleanup runs immediately, then respects its own interval."""
    storage = SQLiteStorage(str(tmp_path / "cleanup.db"))
    mailbox = TaskMailboxManager(storage=storage)
    custom_settings = ServerSettings(
        enable_task_cleanup=True,
        task_retention_hours=1,
        task_cleanup_interval_seconds=3600,
    )
    cleanup = TaskCleanupService(mailbox=mailbox, server_settings=custom_settings)

    first = mailbox._get_or_create("expired-first")
    first.state = TaskState.COMPLETED
    first.updated_at = "2020-01-01T00:00:00+00:00"
    storage.save_task(first)

    result = cleanup.run_cleanup_once()
    assert result["tasks_expired"] == 1
    assert mailbox.get_task("expired-first") is None
    assert cleanup.get_stats()["tasks_expired_total"] == 1
    assert cleanup.get_stats()["last_cleanup_at"] is not None

    second = mailbox._get_or_create("expired-second")
    second.state = TaskState.FAILED
    second.updated_at = "2020-01-01T00:00:00+00:00"
    storage.save_task(second)

    # A regular sweep before interval elapsed must not clean tasks
    assert cleanup.run_cleanup_once()["tasks_expired"] == 0
    assert mailbox.get_task("expired-second") is not None

    cleanup._last_cleanup_monotonic -= 3600
    assert cleanup.run_cleanup_once()["tasks_expired"] == 1
    assert mailbox.get_task("expired-second") is None
    storage.close()
