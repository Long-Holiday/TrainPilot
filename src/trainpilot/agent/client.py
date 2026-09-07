"""Lightweight TrainPilot GPU Agent Client using only Python standard library and requests."""

import logging
import math
import os
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger("trainpilot.agent.client")


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NaN/Inf (float, Decimal, numpy, torch) into JSON-compliant values.

    Also normalizes numpy arrays/tensors, sets, bytes so `requests` never crashes
    on real training metrics (numpy.float32, torch.Tensor, etc.).
    """
    # Native float fast path
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    # Bool/None/int/str pass through (bool must precede int check semantics, but both are JSON-safe)
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    # Decimal (no hard dependency on value, duck-typed to avoid import cost when unused)
    try:
        import decimal as _decimal
        if isinstance(obj, _decimal.Decimal):
            if obj.is_nan():
                return "NaN"
            if obj.is_infinite():
                return "Infinity" if obj > 0 else "-Infinity"
            return float(obj)
    except Exception:
        pass
    # numpy scalars / arrays (optional dependency, duck-typed)
    try:
        import numpy as _np  # type: ignore
        if isinstance(obj, _np.generic):
            return _sanitize_for_json(obj.item())
        if isinstance(obj, _np.ndarray):
            return _sanitize_for_json(obj.tolist())
    except ImportError:
        pass
    except Exception:
        pass
    # torch tensors (optional): 0-dim -> scalar, else -> list
    try:
        _mod = type(obj).__module__.split(".")[0]
        if _mod == "torch" and hasattr(obj, "numel"):
            try:
                _any: Any = obj
                if _any.numel() == 1:
                    try:
                        return _sanitize_for_json(_any.item())
                    except Exception:
                        pass
                if hasattr(_any, "tolist"):
                    return _sanitize_for_json(_any.tolist())
                if hasattr(_any, "detach"):
                    return _sanitize_for_json(_any.detach().cpu().tolist())
            except Exception:
                return str(obj)
    except Exception:
        pass
    # Containers
    if isinstance(obj, dict):
        return {str(k) if not isinstance(k, str) else k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return repr(obj)
    return obj


def _coerce_to_float(value: Any) -> float:
    """Best-effort conversion of loss-like values (Tensor/numpy/str) to float."""
    if hasattr(value, "item"):
        try:
            # torch 0-dim tensor / numpy scalar
            v = value.item()
            if isinstance(v, (int, float)):
                return float(v)
            value = v
        except Exception:
            pass
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("nan", "+nan", "-nan"):
            return float("nan")
        if s in ("inf", "+inf", "infinity", "+infinity"):
            return float("inf")
        if s in ("-inf", "-infinity"):
            return float("-inf")
        return float(value)
    return float(value)


class TrainPilotClient:
    """Zero-credential, lightweight HTTP client communicating with TrainPilot Gateway.

    网关地址解析 (GPU 侧):
    显式 ``gateway_url`` 参数 > ``$TRAINPILOT_GATEWAY_URL`` >
    ``http://$TRAINPILOT_HOST:$TRAINPILOT_PORT`` > 默认 ``http://127.0.0.1:28780``。
    因此 GPU 机器只需设置 ``TRAINPILOT_HOST=<Web 公网 IP/域名>`` 即可, 无需拼完整 URL。
    """

    def __init__(
        self,
        gateway_url: Optional[str] = None,
        task_id: Optional[str] = None,
        timeout: int = 10,
        api_token: Optional[str] = None,
        enable_offline_buffering: bool = True,
        max_offline_buffer_size: int = 1000,
    ):
        if gateway_url is None:
            # 延迟导入避免循环依赖; 解析逻辑收敛到 common.gateway
            from trainpilot.common.gateway import resolve_gateway_url

            gateway_url = resolve_gateway_url()
        if task_id is None:
            task_id = os.environ.get("TRAINPILOT_TASK_ID", "train-task-default")
        if api_token is None:
            api_token = os.environ.get("TRAINPILOT_API_TOKEN")
        self.gateway_url = gateway_url.rstrip("/")
        self.task_id = task_id
        self.timeout = timeout
        self.api_token = api_token
        self.enable_offline_buffering = enable_offline_buffering
        self.max_offline_buffer_size = max(1, int(max_offline_buffer_size))
        self._offline_buffer: deque = deque(maxlen=self.max_offline_buffer_size)
        self._buffer_lock = threading.Lock()
        self.session = requests.Session()
        if api_token:
            self.session.headers.update({"Authorization": f"Bearer {api_token}"})

    def get_buffered_count(self) -> int:
        """Return the number of events currently buffered offline."""
        with self._buffer_lock:
            return len(self._offline_buffer)

    def flush_offline_buffer(self) -> int:
        """Attempt to flush buffered offline events in chronological order to the gateway.

        Stops at the first network failure to preserve strict causal ordering.
        Returns the number of successfully delivered events.
        """
        flushed_count = 0
        url = f"{self.gateway_url}/api/tasks/notify"
        while True:
            with self._buffer_lock:
                if not self._offline_buffer:
                    break
                payload = self._offline_buffer[0]

            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                with self._buffer_lock:
                    if self._offline_buffer and self._offline_buffer[0] is payload:
                        self._offline_buffer.popleft()
                flushed_count += 1
            except requests.RequestException as exc:
                logger.debug("[%s] Offline buffer flush paused: %s", self.task_id, exc)
                break

        if flushed_count > 0:
            logger.info("[%s] Flushed %d buffered offline events to gateway", self.task_id, flushed_count)
        return flushed_count

    def notify_event(
        self,
        event_type: str,
        message: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        agent_note: Optional[str] = None,
        fail_silently: Optional[bool] = None,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Report any event (alert, milestone, completed, failed) to the gateway.

        Resilience features:
        - Automatic offline buffering: network outages buffer events instead of crashing training.
        - Auto-flush: draining pending offline events when network recovers.
        - Exponential backoff retry: for critical alerts and states.
        """
        is_routine = event_type.lower() in ("milestone", "heartbeat")
        if fail_silently is None:
            fail_silently = is_routine
        if max_retries is None:
            max_retries = 0 if is_routine else 3

        # Drain queued offline buffer if online
        if self.enable_offline_buffering and self.get_buffered_count() > 0:
            try:
                self.flush_offline_buffer()
            except Exception:
                pass

        url = f"{self.gateway_url}/api/tasks/notify"
        raw_payload = {
            "task_id": self.task_id,
            "event_type": event_type,
            "message": message,
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "extra": extra,
            "agent_note": agent_note,
        }
        payload = _sanitize_for_json(raw_payload)

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            if attempt > 0:
                backoff = min(6.0, 0.4 * (2 ** (attempt - 1)))
                time.sleep(backoff)
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                logger.info("[%s] Event '%s' reported successfully: %s", self.task_id, event_type, message)
                return data
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.warning("[%s] Retry %d/%d reporting '%s' due to network error: %s",
                                   self.task_id, attempt + 1, max_retries, event_type, exc)

        # All attempts failed: buffer offline or raise
        if self.enable_offline_buffering and fail_silently:
            with self._buffer_lock:
                self._offline_buffer.append(payload)
                buf_size = len(self._offline_buffer)
            logger.warning("[%s] Network unavailable; buffered '%s' event (queue: %d). Error: %s",
                           self.task_id, event_type, buf_size, last_exc)
            return {
                "success": True,
                "buffered": True,
                "task_id": self.task_id,
                "event_type": event_type,
                "message": f"Buffered offline ({buf_size} pending)",
            }

        logger.error("[%s] Failed to report event '%s': %s", self.task_id, event_type, last_exc)
        raise last_exc

    def notify_milestone(
        self,
        message: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        agent_note: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        fail_silently: Optional[bool] = None,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Report a cruising milestone (e.g., epoch finished, checkpoint saved).

        ``agent_note``: 外部 AI 智能体针对当前实际情况自主生成的 1-3 句点评,
        将展示在飞书卡片 ``🤖 Agent 智能点评`` 区块。如不提供则仅展示 message
        (向后兼容)。
        """
        return self.notify_event(
            event_type="milestone",
            message=message,
            step=step,
            epoch=epoch,
            metrics=metrics,
            extra=extra,
            agent_note=agent_note,
            fail_silently=fail_silently,
            max_retries=max_retries,
        )

    def notify_alert(
        self,
        message: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        agent_note: Optional[str] = None,
        fail_silently: Optional[bool] = None,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Report a high-priority alert (Loss NaN, OOM, loss explosion) and request HITL intervention."""
        return self.notify_event(
            event_type="alert",
            message=message,
            step=step,
            epoch=epoch,
            metrics=metrics,
            extra=extra,
            agent_note=agent_note,
            fail_silently=fail_silently,
            max_retries=max_retries,
        )

    def poll_instruction(
        self,
        timeout: Optional[float] = 300.0,
        interval: float = 2.0,
        pop: bool = True,
        long_poll: bool = True,
        long_poll_timeout: float = 20.0,
    ) -> Dict[str, Any]:
        """Poll the mailbox until a human decision instruction is available or timeout occurs.

        By default enables HTTP Long Polling (long_poll=True), holding the connection on
        the server for up to `long_poll_timeout` seconds, achieving sub-second reaction
        speed upon human click while cutting network traffic by >90%.
        """
        url = f"{self.gateway_url}/api/tasks/{self.task_id}/instruction"
        start_time = time.time()
        mode_desc = f"long-polling every {long_poll_timeout:.0f}s" if long_poll else f"short-polling every {interval:.1f}s"
        logger.info("[%s] Waiting for human-in-the-loop decision (%s)...", self.task_id, mode_desc)

        while True:
            elapsed = time.time() - start_time
            if timeout is not None and elapsed >= timeout:
                raise TimeoutError(f"Timed out after {timeout} seconds waiting for decision on task {self.task_id}")

            params = {"pop": str(pop).lower()}
            req_timeout = self.timeout

            if long_poll:
                remaining = (timeout - elapsed) if timeout is not None else long_poll_timeout
                current_lp_wait = max(1.0, min(long_poll_timeout, remaining))
                params["wait_timeout"] = str(round(current_lp_wait, 1))
                req_timeout = current_lp_wait + self.timeout

            try:
                resp = self.session.get(url, params=params, timeout=req_timeout)
                resp.raise_for_status()
                data = resp.json()
                if data.get("ready"):
                    logger.info("[%s] Decision received: action='%s' by %s",
                                self.task_id, data.get("action"), data.get("decision_by"))
                    return data
                # If long polling returned without ready, continue to next long poll cycle immediately
                if not long_poll:
                    time.sleep(interval)
                else:
                    time.sleep(0.1)
            except requests.RequestException as exc:
                logger.warning("[%s] Network error while polling instruction: %s", self.task_id, exc)
                time.sleep(interval)

    def ack_instruction(
        self,
        action: str,
        instruction_id: Optional[str] = None,
        status: str = "success",
        message: Optional[str] = None,
        solution: Optional[str] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Acknowledge completion of a recovery instruction to transition state back to RUNNING."""
        url = f"{self.gateway_url}/api/tasks/{self.task_id}/ack"
        raw_payload = {
            "task_id": self.task_id,
            "instruction_id": instruction_id,
            "action": action,
            "status": status,
            "message": message,
            "solution": solution,
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
        }
        payload = _sanitize_for_json(raw_payload)
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            if attempt > 0:
                backoff = min(6.0, 0.5 * (2 ** (attempt - 1)))
                time.sleep(backoff)
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                logger.info("[%s] Instruction '%s' acked with status '%s'", self.task_id, action, status)
                return data
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.warning("[%s] Retry %d/%d acknowledging instruction '%s' due to error: %s",
                                   self.task_id, attempt + 1, max_retries, action, exc)

        logger.error("[%s] Failed to acknowledge instruction: %s", self.task_id, last_exc)
        raise last_exc

    def send_heartbeat(
        self,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send a lightweight heartbeat ping and flush any buffered offline events."""
        url = f"{self.gateway_url}/api/tasks/{self.task_id}/heartbeat"
        raw_payload = {
            "task_id": self.task_id,
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
        }
        payload = _sanitize_for_json(raw_payload)
        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                logger.warning("[%s] Heartbeat rejected: HTTP %s %s", self.task_id, resp.status_code, resp.text[:200])
                return False
            # If heartbeat succeeds, network is healthy; drain offline buffer
            if self.enable_offline_buffering and self.get_buffered_count() > 0:
                try:
                    self.flush_offline_buffer()
                except Exception:
                    pass
            return True
        except requests.RequestException as exc:
            logger.warning("[%s] Heartbeat delivery failed: %s", self.task_id, exc)
            return False

    def get_status(self) -> Dict[str, Any]:
        """Fetch current status and metadata of the task from the gateway."""
        url = f"{self.gateway_url}/api/tasks/{self.task_id}/status"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
