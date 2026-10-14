"""Automated tests for start.sh (TrainPilot Control Plane & MCP Server startup script)."""

import os
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
START_SH = os.path.join(PROJECT_ROOT, "start.sh")


def _start_sh(*args):
    return subprocess.run([START_SH, *args], cwd=PROJECT_ROOT, capture_output=True, text=True)


def test_start_script_help():
    """--help documents the MCP Server, the daemon flags and the host/port overrides."""
    res = _start_sh("--help")
    assert res.returncode == 0
    for expected in (
        "MCP Server",
        "--daemon",
        "28780",
        "-p <PORT>",
        "--host <HOST>",
        "TRAINPILOT_BIND_HOST",
        "/mcp",
    ):
        assert expected in res.stdout, expected
    assert "setup_skills.sh" not in res.stdout


def test_start_script_status_when_not_running():
    """Verify ./start.sh --status correctly reports when service is stopped."""
    res = _start_sh("--status")
    assert res.returncode == 0
    assert "未运行" in res.stdout
