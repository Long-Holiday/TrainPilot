"""Tests for the TrainPilot Agent Skill CLI tool."""

import json
import subprocess
import sys
import threading
import time
import pytest
import uvicorn

from trainpilot.server.main import app
from trainpilot.server.mailbox import default_mailbox

TEST_HOST = "127.0.0.1"
TEST_PORT = 18923
BASE_URL = f"http://{TEST_HOST}:{TEST_PORT}"


class UvicornTestServer(uvicorn.Server):
    """Run uvicorn in a background thread for testing."""

    def install_signal_handlers(self):
        pass


@pytest.fixture(scope="module")
def live_server():
    config = uvicorn.Config(app=app, host=TEST_HOST, port=TEST_PORT, log_level="warning")
    server = UvicornTestServer(config=config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to become responsive
    time.sleep(1.0)
    yield BASE_URL
    server.should_exit = True
    thread.join(timeout=2)


def run_cli_cmd(*args):
    cli_path = "skills/trainpilot/scripts/trainpilot_tool.py"
    cmd = [sys.executable, cli_path, "--gateway", BASE_URL, *args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res


def test_skills_cli_full_flow(live_server):
    task_id = "skill-cli-test-task"
    default_mailbox.reset()

    # 1. Report Milestone via CLI
    res = run_cli_cmd(
        "--task-id", task_id,
        "report-milestone",
        "--message", "Epoch 1 finished",
        "--step", "50",
        "--metrics", "loss=0.45,acc=0.88",
    )
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["success"] is True
    assert out["state"] == "RUNNING"

    # 2. Report Alert via CLI
    res = run_cli_cmd(
        "--task-id", task_id,
        "report-alert",
        "--message", "Loss NaN at step 51",
        "--step", "51",
    )
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["state"] == "WAITING"

    # 3. Poll without wait -> ready: false
    res = run_cli_cmd(
        "--task-id", task_id,
        "poll-instruction",
    )
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["ready"] is False
    assert out["status"] == "WAITING"

    # 4. Mock human decision via CLI
    res = run_cli_cmd(
        "--task-id", task_id,
        "mock-decision",
        "--action", "reduce_lr_rollback",
        "--operator", "Agent-Tester",
    )
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["action"] == "reduce_lr_rollback"
    inst_id = out["instruction_id"]

    # 5. Poll with pop -> ready: true
    res = run_cli_cmd(
        "--task-id", task_id,
        "poll-instruction",
    )
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["ready"] is True
    assert out["action"] == "reduce_lr_rollback"

    # 6. Ack instruction via CLI
    res = run_cli_cmd(
        "--task-id", task_id,
        "ack-instruction",
        "--action", "reduce_lr_rollback",
        "--instruction-id", inst_id,
        "--status", "success",
    )
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["state"] == "RUNNING"

    # 7. Query status via CLI
    res = run_cli_cmd(
        "--task-id", task_id,
        "get-status",
    )
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["state"] == "RUNNING"
    assert out["task_id"] == task_id
