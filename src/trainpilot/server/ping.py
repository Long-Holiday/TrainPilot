"""ICMP ping helper for GPU liveness checks.

看门狗每轮 ping 任务上报的 ``gpu_host``: 能 ping 通即视为 GPU 服务器存活,
ping 不通或从未上报过 IP 则视为失联并告警。Agent 侧无需定时发送心跳。
"""

import logging
import shutil
import socket
import subprocess

logger = logging.getLogger("trainpilot.ping")


def normalize_ping_host(host: str | None) -> str | None:
    """去掉 scheme/路径/端口, 取出可 ping 的主机部分。"""
    if not host:
        return None
    h = host.strip().strip("'\"")
    if not h:
        return None
    for prefix in ("http://", "https://"):
        if h.lower().startswith(prefix):
            h = h[len(prefix):]
            break
    h = h.split("/")[0].strip()
    h = h.split("@")[-1].strip()
    if h.startswith("["):
        end = h.find("]")
        if end != -1:
            return h[1:end] or None
        return None
    if h.count(":") == 1:
        maybe_host, maybe_port = h.rsplit(":", 1)
        if maybe_port.isdigit() and maybe_host:
            h = maybe_host
    return h or None


def _tcp_fallback(host: str, timeout_seconds: float) -> bool:
    """无 ping 命令时的降级探测: 尝试常见端口, 任一连通即视为存活。"""
    for port in (22, 80, 443):
        try:
            with socket.create_connection((host, port), timeout=timeout_seconds):
                return True
        except Exception:
            continue
    return False


def ping_host(host: str | None, timeout_seconds: float = 3.0) -> bool:
    """Ping 一个 GPU 主机, 返回是否可达。永不抛异常(失败即 False)。"""
    target = normalize_ping_host(host)
    if not target:
        return False
    timeout_seconds = max(0.5, float(timeout_seconds or 3.0))
    ping_bin = shutil.which("ping")
    if ping_bin is None:
        return _tcp_fallback(target, min(timeout_seconds, 3.0))
    try:
        import platform

        system = platform.system().lower()
        if system == "windows":
            timeout_ms = str(max(500, int(timeout_seconds * 1000)))
            cmd = [ping_bin, "-n", "1", "-w", timeout_ms, target]
        elif system == "darwin":
            timeout_ms = str(max(500, int(timeout_seconds * 1000)))
            cmd = [ping_bin, "-c", "1", "-W", timeout_ms, target]
        else:
            cmd = [ping_bin, "-c", "1", "-W", str(max(1, int(timeout_seconds))), target]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds + 2.0,
        )
        return proc.returncode == 0
    except Exception as exc:
        logger.debug("ping %s failed: %s", target, exc)
        return False
