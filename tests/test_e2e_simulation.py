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
    config = uvicorn.Config(app=app, host=TEST_HOST, port=TEST_PORT, log_level="warning")
    server = UvicornTestServer(config=config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    time.sleep(1.0)
    yield BASE_URL
    server.should_exit = True
    thread.join(timeout=2)


def test_mock_training_e2e_run(live_server_e2e):
    default_mailbox.reset()
    task_id = "e2e-simulation-task"

    # Run simulated training which triggers NaN at step 6 and auto-recovers after 3 seconds
    run_mock_training(gateway_url=live_server_e2e, task_id=task_id)

    # Check final task state
    summary = default_mailbox.get_task(task_id)
    assert summary is not None
    assert summary.state.value == "COMPLETED"
    assert summary.events_count >= 3  # milestone + alert + completed
