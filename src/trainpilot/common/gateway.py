"""Unified gateway address resolution for Web (Control Plane) vs GPU (Agent) sides.

约定 (v0.2+):
- Web 服务器 (Control Plane): 只关心 **绑定地址** ``TRAINPILOT_BIND_HOST`` /
  ``TRAINPILOT_HOST`` + ``TRAINPILOT_PORT``。默认绑定 ``0.0.0.0``。
- GPU 服务器 (Agent): 只关心 **网关访问地址**。优先级:
  ``TRAINPILOT_GATEWAY_URL`` (完整 URL, 最高) >
  ``http://<TRAINPILOT_HOST>:<TRAINPILOT_PORT>`` >
  默认 ``http://127.0.0.1:28780``。

``TRAINPILOT_HOST`` 因此在两侧含义不同, 必须按机器分别配置:
- Web 侧示例: ``TRAINPILOT_HOST=0.0.0.0`` (或留空取默认)
- GPU 侧示例: ``TRAINPILOT_HOST=35.202.16.245`` (Web 公网 IP/域名, 不要写 0.0.0.0)

本模块被 server (bind 解析) 与 agent/client (网关 URL 解析) 共同复用,
``skills/trainpilot/scripts/trainpilot_tool.py`` 因需保持单文件零依赖可拷贝,
在该脚本内保留了一份精简副本 (逻辑与此处保持一致, 修改时请同步)。
"""

import os

DEFAULT_PORT = 28780
DEFAULT_GATEWAY_URL = "http://127.0.0.1:28780"

# 可直接用于 uvicorn --host 的本地绑定值
_LOCAL_BIND_HOSTS = {"", "0.0.0.0", "::", "127.0.0.1", "localhost"}


def _normalize_host(host: str | None) -> str | None:
    if host is None:
        return None
    h = host.strip().strip("'\"")
    # 兼容用户误填 scheme / 路径 / 端口, 如 "http://1.2.3.4:28780/" -> "1.2.3.4"
    for prefix in ("http://", "https://"):
        if h.lower().startswith(prefix):
            h = h[len(prefix):]
            break
    h = h.split("/")[0].strip()
    # 去掉末尾 ":port" (小心 IPv6 带括号场景, 仅处理最常见的 IPv4/hostname 情况)
    if h.count(":") == 1 and not h.startswith("["):
        maybe_host, maybe_port = h.rsplit(":", 1)
        if maybe_port.isdigit() and maybe_host:
            h = maybe_host
    return h or None


def _normalize_port(port: str | int | None) -> int | None:
    if port is None:
        return None
    try:
        p = int(str(port).strip())
        if 1 <= p <= 65535:
            return p
    except (ValueError, TypeError):
        pass
    return None


def resolve_bind_host(
    explicit: str | None = None,
    *,
    bind_host_env: str | None = None,
    host_env: str | None = None,
) -> str:
    """解析 Web 侧 uvicorn 绑定地址。

    优先级: explicit (--host 参数) > $TRAINPILOT_BIND_HOST >
    $TRAINPILOT_HOST (仅当其本身就是本地绑定值时) > "0.0.0.0"。
    这样即使误把 GPU 侧的公网 IP 拷贝到 Web 侧, 也不会导致 bind 失败。
    """
    if bind_host_env is None:
        bind_host_env = os.environ.get("TRAINPILOT_BIND_HOST")
    if host_env is None:
        host_env = os.environ.get("TRAINPILOT_HOST")

    for candidate in (explicit, bind_host_env):
        h = _normalize_host(candidate) if candidate is not None else None
        if h:
            return h

    h = _normalize_host(host_env) if host_env else None
    if h and h in _LOCAL_BIND_HOSTS:
        return h
    return "0.0.0.0"


def resolve_public_host(host_env: str | None = None) -> str:
    """解析对外展示/拼接用的网关 Host (0.0.0.0 -> 127.0.0.1, 空 -> 127.0.0.1)。"""
    if host_env is None:
        host_env = os.environ.get("TRAINPILOT_HOST")
    h = _normalize_host(host_env) if host_env else None
    if not h or h in ("0.0.0.0", "::"):
        return "127.0.0.1"
    return h


def resolve_gateway_url(
    gateway_url: str | None = None,
    *,
    host: str | None = None,
    port: str | int | None = None,
) -> str:
    """解析 GPU 侧网关访问地址 (Agent/Skill/Example 通用)。

    优先级:
    1. 显式传入的 ``gateway_url`` 参数 (非空即用)
    2. 环境变量 ``TRAINPILOT_GATEWAY_URL`` (完整 URL, 去尾斜杠)
    3. ``http://<TRAINPILOT_HOST>:<TRAINPILOT_PORT>`` 拼接
       (HOST 为空/0.0.0.0 时自动折叠为 127.0.0.1)
    4. 默认 ``http://127.0.0.1:28780``
    """
    if gateway_url and str(gateway_url).strip():
        return str(gateway_url).strip().rstrip("/")

    env_url = os.environ.get("TRAINPILOT_GATEWAY_URL")
    if env_url and env_url.strip():
        return env_url.strip().rstrip("/")

    raw_host = host if host is not None else os.environ.get("TRAINPILOT_HOST")
    raw_port = port if port is not None else os.environ.get("TRAINPILOT_PORT")
    h = _normalize_host(raw_host) if raw_host else None
    if h:
        if h in ("0.0.0.0", "::"):
            h = "127.0.0.1"
        p = _normalize_port(raw_port) or _normalize_port(os.environ.get("TRAINPILOT_PORT")) or DEFAULT_PORT
        return f"http://{h}:{p}"

    # HOST 未设置但 PORT 设置了 (Web 本地调试常见): 仍给出可用本地地址
    p_only = _normalize_port(raw_port)
    if p_only:
        return f"http://127.0.0.1:{p_only}"

    return DEFAULT_GATEWAY_URL


def resolve_port(explicit: str | int | None = None) -> int:
    """解析监听/连接端口: explicit > $TRAINPILOT_PORT > 28780。"""
    p = _normalize_port(explicit) if explicit is not None else None
    if p:
        return p
    return _normalize_port(os.environ.get("TRAINPILOT_PORT")) or DEFAULT_PORT
