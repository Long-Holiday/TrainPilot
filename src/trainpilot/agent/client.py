"""Lightweight TrainPilot GPU Agent Client using only Python standard library and requests."""

import logging
import math
import os
import time
from typing import Any, Dict, Optional
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
        self.session = requests.Session()
        if api_token:
            self.session.headers.update({"Authorization": f"Bearer {api_token}"})

    def notify_event(
        self,
        event_type: str,
        message: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        agent_note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Report any event (alert, milestone, completed, failed) to the gateway."""
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
        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            logger.info("[%s] Event '%s' reported successfully: %s", self.task_id, event_type, message)
            return data
        except requests.RequestException as exc:
            logger.error("[%s] Failed to report event '%s': %s", self.task_id, event_type, exc)
            raise

    def notify_milestone(
        self,
        message: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        agent_note: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
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
        )

    def notify_alert(
        self,
        message: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        agent_note: Optional[str] = None,
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
        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            logger.info("[%s] Instruction '%s' acked with status '%s'", self.task_id, action, status)
            return data
        except requests.RequestException as exc:
            logger.error("[%s] Failed to acknowledge instruction: %s", self.task_id, exc)
            raise

    def send_heartbeat(
        self,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send a lightweight heartbeat ping."""
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
