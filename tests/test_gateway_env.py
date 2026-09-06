"""Tests for Web/GPU TRAINPILOT_HOST distinction and gateway resolution."""

import os
import subprocess

import pytest

from trainpilot.common.gateway import (
    resolve_bind_host,
    resolve_gateway_url,
    resolve_port,
    resolve_public_host,
)
from trainpilot.server.config import ServerSettings

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
START_SH = os.path.join(PROJECT_ROOT, "start.sh")


@pytest.fixture(autouse=True)
def clean_trainpilot_env(monkeypatch):
    for k in ("TRAINPILOT_GATEWAY_URL", "TRAINPILOT_HOST", "TRAINPILOT_BIND_HOST", "TRAINPILOT_PORT"):
        monkeypatch.delenv(k, raising=False)


def test_resolve_gateway_url_prefers_full_url(monkeypatch):
    monkeypatch.setenv("TRAINPILOT_GATEWAY_URL", "http://web.example.com:28780/")
    monkeypatch.setenv("TRAINPILOT_HOST", "1.2.3.4")
    assert resolve_gateway_url() == "http://web.example.com:28780"


def test_resolve_gateway_url_from_host_port_gpu(monkeypatch):
    # GPU 侧: 只填 HOST(公网 IP) + PORT 即可拼接
    monkeypatch.setenv("TRAINPILOT_HOST", "35.202.16.245")
    monkeypatch.setenv("TRAINPILOT_PORT", "28780")
    assert resolve_gateway_url() == "http://35.202.16.245:28780"


def test_resolve_gateway_url_zero_host_folds_to_loopback(monkeypatch):
    # Web 侧 HOST=0.0.0.0 不可直接作为访问地址, 自动折叠为 127.0.0.1
    monkeypatch.setenv("TRAINPILOT_HOST", "0.0.0.0")
    monkeypatch.setenv("TRAINPILOT_PORT", "29580")
    assert resolve_gateway_url() == "http://127.0.0.1:29580"


def test_resolve_gateway_url_default(monkeypatch):
    assert resolve_gateway_url() == "http://127.0.0.1:28780"


def test_resolve_gateway_url_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv("TRAINPILOT_GATEWAY_URL", "http://env-host:28780")
    assert resolve_gateway_url("http://arg-host:9999") == "http://arg-host:9999"


def test_resolve_bind_host_priority(monkeypatch):
    # --host 参数 > BIND_HOST > HOST(本地值) > 0.0.0.0
    monkeypatch.setenv("TRAINPILOT_BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("TRAINPILOT_HOST", "0.0.0.0")
    assert resolve_bind_host(explicit="192.168.1.10") == "192.168.1.10"
    assert resolve_bind_host() == "127.0.0.1"


def test_resolve_bind_host_ignores_public_ip(monkeypatch):
    # 误把 GPU 侧公网 IP 拷贝到 Web 侧时, 必须回退 0.0.0.0 而非 bind 公网 IP
    monkeypatch.setenv("TRAINPILOT_HOST", "35.202.16.245")
    assert resolve_bind_host() == "0.0.0.0"
    monkeypatch.setenv("TRAINPILOT_BIND_HOST", "0.0.0.0")
    assert resolve_bind_host() == "0.0.0.0"


def test_resolve_public_host_folds(monkeypatch):
    monkeypatch.setenv("TRAINPILOT_HOST", "0.0.0.0")
    assert resolve_public_host() == "127.0.0.1"
    monkeypatch.setenv("TRAINPILOT_HOST", "35.202.16.245")
    assert resolve_public_host() == "35.202.16.245"


def test_server_settings_effective_bind_host():
    s = ServerSettings(host="35.202.16.245", port=28780)
    # 公网 IP 不可作为 bind, 自动回退
    assert s.effective_bind_host == "0.0.0.0"
    s2 = ServerSettings(host="0.0.0.0", bind_host="127.0.0.1", port=28780)
    assert s2.effective_bind_host == "127.0.0.1"
    s3 = ServerSettings(host="0.0.0.0", port=29580)
    assert s3.public_host == "127.0.0.1"
    assert s3.advertised_gateway_url == "http://127.0.0.1:29580"


def test_client_uses_host_fallback(monkeypatch):
    from trainpilot.agent.client import TrainPilotClient

    monkeypatch.setenv("TRAINPILOT_HOST", "35.202.16.245")
    monkeypatch.setenv("TRAINPILOT_PORT", "28780")
    c = TrainPilotClient(task_id="t1")
    assert c.gateway_url == "http://35.202.16.245:28780"


def test_start_script_clean_flags():
    res = subprocess.run(
        [START_SH, "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "TRAINPILOT_BIND_HOST" in res.stdout
