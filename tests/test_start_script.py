"""Automated tests for start.sh (TrainPilot Control Plane & MCP Server startup script)."""

import os
import subprocess
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
START_SH = os.path.join(PROJECT_ROOT, "start.sh")


def test_start_script_help():
    """Verify start.sh --help outputs usage instructions for MCP Server."""
    res = subprocess.run([START_SH, "--help"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert res.returncode == 0
    assert "MCP Server" in res.stdout
    assert "--daemon" in res.stdout
    assert "28780" in res.stdout
    assert "--host" in res.stdout
    assert "/sse" in res.stdout
    assert "setup_skills.sh" not in res.stdout


def test_start_script_status_when_not_running():
    """Verify ./start.sh --status correctly reports when service is stopped."""
    res = subprocess.run([START_SH, "--status"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert res.returncode == 0
    assert "未运行" in res.stdout


def test_start_script_custom_port_flag():
    """Verify start.sh accepts custom port and host arguments."""
    res = subprocess.run([START_SH, "--help"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert "-p <PORT>" in res.stdout
    assert "--host <HOST>" in res.stdout
