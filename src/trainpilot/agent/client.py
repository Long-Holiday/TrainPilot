"""Lightweight TrainPilot GPU Agent Client using only Python standard library and requests."""

import logging
import math
import time
from typing import Any, Dict, Optional
import requests

logger = logging.getLogger("trainpilot.agent.client")


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert float('nan') and float('inf') into JSON-compliant representations."""
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    return obj


class TrainPilotClient:
    """Zero-credential, lightweight HTTP client communicating with TrainPilot Gateway."""

    def __init__(
        self,
        gateway_url: str = "http://localhost:8000",
        task_id: str = "train-task-default",
        timeout: int = 10,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.task_id = task_id
        self.timeout = timeout
        self.session = requests.Session()

    def notify_event(
        self,
        event_type: str,
        message: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
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
    ) -> Dict[str, Any]:
        """Report a cruising milestone (e.g., epoch finished, checkpoint saved)."""
        return self.notify_event(
            event_type="milestone",
            message=message,
            step=step,
            epoch=epoch,
            metrics=metrics,
        )

    def notify_alert(
        self,
        message: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Report a high-priority alert (Loss NaN, OOM, loss explosion) and request HITL intervention."""
        return self.notify_event(
            event_type="alert",
            message=message,
            step=step,
            epoch=epoch,
            metrics=metrics,
            extra=extra,
        )

    def poll_instruction(
        self,
        timeout: Optional[float] = 300.0,
        interval: float = 3.0,
        pop: bool = True,
    ) -> Dict[str, Any]:
        """Poll the mailbox until a human decision instruction is available or timeout occurs."""
        url = f"{self.gateway_url}/api/tasks/{self.task_id}/instruction"
        start_time = time.time()
        logger.info("[%s] Waiting for human-in-the-loop decision (polling every %.1fs)...", self.task_id, interval)

        while True:
            try:
                resp = self.session.get(url, params={"pop": str(pop).lower()}, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if data.get("ready"):
                    logger.info("[%s] Decision received: action='%s' by %s",
                                self.task_id, data.get("action"), data.get("decision_by"))
                    return data
            except requests.RequestException as exc:
                logger.warning("[%s] Network error while polling instruction: %s", self.task_id, exc)

            if timeout is not None and (time.time() - start_time) >= timeout:
                raise TimeoutError(f"Timed out after {timeout} seconds waiting for decision on task {self.task_id}")

            time.sleep(interval)

    def ack_instruction(
        self,
        action: str,
        instruction_id: Optional[str] = None,
        status: str = "success",
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Acknowledge completion of a recovery instruction to transition state back to RUNNING."""
        url = f"{self.gateway_url}/api/tasks/{self.task_id}/ack"
        payload = {
            "task_id": self.task_id,
            "instruction_id": instruction_id,
            "action": action,
            "status": status,
            "message": message,
        }
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
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Fetch current status and metadata of the task from the gateway."""
        url = f"{self.gateway_url}/api/tasks/{self.task_id}/status"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
