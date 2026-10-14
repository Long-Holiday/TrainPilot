"""Tests for Web/GPU TRAINPILOT_HOST distinction and gateway resolution."""

import pytest

from trainpilot.common.gateway import (
    resolve_bind_host,
    resolve_gateway_url,
    resolve_public_host,
)
from trainpilot.server.config import ServerSettings


@pytest.fixture(autouse=True)
def clean_trainpilot_env(monkeypatch):
    for k in ("TRAINPILOT_GATEWAY_URL", "TRAINPILOT_HOST", "TRAINPILOT_BIND_HOST", "TRAINPILOT_PORT"):
        monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize("env,expected", [
    # GPU 侧: 只填 HOST(公网 IP) + PORT 即可拼接
    ({"TRAINPILOT_HOST": "35.202.16.245", "TRAINPILOT_PORT": "28780"},
     "http://35.202.16.245:28780"),
    # Web 侧 HOST=0.0.0.0 不可直接作为访问地址, 自动折叠为 127.0.0.1
    ({"TRAINPILOT_HOST": "0.0.0.0", "TRAINPILOT_PORT": "29580"},
     "http://127.0.0.1:29580"),
    ({}, "http://127.0.0.1:28780"),
])
def test_resolve_gateway_url(monkeypatch, env, expected):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert resolve_gateway_url() == expected


def test_resolve_gateway_url_precedence(monkeypatch):
    """GATEWAY_URL beats HOST/PORT, and an explicit argument beats both."""
    monkeypatch.setenv("TRAINPILOT_GATEWAY_URL", "http://web.example.com:28780/")
    monkeypatch.setenv("TRAINPILOT_HOST", "1.2.3.4")
    assert resolve_gateway_url() == "http://web.example.com:28780"

    monkeypatch.setenv("TRAINPILOT_GATEWAY_URL", "http://env-host:28780")
    assert resolve_gateway_url("http://arg-host:9999") == "http://arg-host:9999"


@pytest.mark.parametrize("env,expected", [
    # --host 参数 > BIND_HOST > HOST(本地值) > 0.0.0.0
    ({"TRAINPILOT_BIND_HOST": "127.0.0.1", "TRAINPILOT_HOST": "0.0.0.0"}, "127.0.0.1"),
    # 误把 GPU 侧公网 IP 拷贝到 Web 侧时, 必须回退 0.0.0.0 而非 bind 公网 IP
    ({"TRAINPILOT_HOST": "35.202.16.245"}, "0.0.0.0"),
    ({"TRAINPILOT_HOST": "0.0.0.0"}, "0.0.0.0"),
])
def test_resolve_bind_host(monkeypatch, env, expected):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert resolve_bind_host() == expected
    # 显式 --host 参数始终优先
    assert resolve_bind_host(explicit="192.168.1.10") == "192.168.1.10"


@pytest.mark.parametrize("host,expected", [
    ("0.0.0.0", "127.0.0.1"),
    ("35.202.16.245", "35.202.16.245"),
])
def test_resolve_public_host(monkeypatch, host, expected):
    monkeypatch.setenv("TRAINPILOT_HOST", host)
    assert resolve_public_host() == expected


def test_server_settings_effective_hosts():
    s = ServerSettings(host="35.202.16.245", port=28780)
    # 公网 IP 不可作为 bind, 自动回退
    assert s.effective_bind_host == "0.0.0.0"
    s2 = ServerSettings(host="0.0.0.0", bind_host="127.0.0.1", port=28780)
    assert s2.effective_bind_host == "127.0.0.1"
    s3 = ServerSettings(host="0.0.0.0", port=29580)
    assert s3.public_host == "127.0.0.1"
    assert s3.advertised_gateway_url == "http://127.0.0.1:29580"
