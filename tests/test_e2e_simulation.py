"""End-to-end integration test simulating training, NaN anomaly, and HITL recovery."""

import threading
import time
import pytest
import uvicorn

from examples.mock_training import run_mock_training
from trainpilot.server.mailbox import default_mailbox
from trainpilot.server.main import app

TEST_HOST = "127.0.0.1"
TEST_PORT = 18925
BASE_URL = f"http://{TEST_HOST}:{TEST_PORT}"


class UvicornTestServer(uvicorn.Server):
    def install_signal_handlers(self):
        pass


@pytest.fixture(scope="module")
def live_server_e2e():
    # Auth is already disabled for tests by tests/conftest.py's global isolation fixture.
    config = uvicorn.Config(app=app, host=TEST_HOST, port=TEST_PORT, log_level="warning")
    server = UvicornTestServer(config=config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    time.sleep(1.0)
    yield BASE_URL
    server.should_exit = True
    thread.join(timeout=2)


def _watch_states(task_id):
    """Sample the task state in the background, collapsing repeats.

    Returns (states, stop): call stop() once the run finishes and read states.
    """
    states = []
    stop = threading.Event()

    def _sample():
        while not stop.is_set():
            summary = default_mailbox.get_task(task_id)
            if summary is not None and (not states or states[-1] != summary.state.value):
                states.append(summary.state.value)
            time.sleep(0.02)

    thread = threading.Thread(target=_sample, daemon=True)
    thread.start()
    return states, lambda: (stop.set(), thread.join(timeout=2))


def test_mock_training_e2e_run(live_server_e2e):
    default_mailbox.reset()
    task_id = "e2e-simulation-task"

    # Run simulated training which triggers NaN at step 6 and recovers after 3 seconds
    states, stop = _watch_states(task_id)
    try:
        run_mock_training(gateway_url=live_server_e2e, task_id=task_id)
    finally:
        stop()

    # Check final task state
    summary = default_mailbox.get_task(task_id)
    assert summary is not None
    assert summary.state.value == "COMPLETED"
    assert summary.events_count >= 3  # milestone + alert + completed

    # The HITL loop must actually be exercised. A run that dropped poll/ack entirely
    # would still end COMPLETED, so assert it went WAITING and came *back* to RUNNING
    # before finishing. (RECOVERING is not asserted: it lasts microseconds between
    # the polling pop and the ack, too short for a sampling watcher to observe.)
    assert states[0] == "RUNNING", states
    assert "WAITING" in states, states
    assert "RUNNING" in states[states.index("WAITING") + 1:], states
