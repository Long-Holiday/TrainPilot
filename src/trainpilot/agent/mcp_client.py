"""Native MCP (Model Context Protocol) Client for TrainPilot GPU Nodes.

Connects from private GPU worker nodes (behind NAT / firewalls) to the public
TrainPilot MCP Server over SSE (Server-Sent Events) or HTTP.
Provides both async and sync methods to invoke control plane tools.
"""

import asyncio
import json
import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger("trainpilot.agent.mcp_client")


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert NaN/Inf (float, Decimal, numpy, torch) into JSON-compliant values.

    Ensures tensor/array values from PyTorch/Numpy do not crash JSON serialization.
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return obj
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
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
    try:
        import numpy as _np
        if isinstance(obj, _np.generic):
            return _sanitize_for_json(obj.item())
        if isinstance(obj, _np.ndarray):
            return _sanitize_for_json(obj.tolist())
    except Exception:
        pass
    try:
        _mod = type(obj).__module__.split(".")[0]
        if _mod == "torch" and hasattr(obj, "numel"):
            if obj.numel() == 1:
                return _sanitize_for_json(obj.item())
            if hasattr(obj, "tolist"):
                return _sanitize_for_json(obj.tolist())
            if hasattr(obj, "detach"):
                return _sanitize_for_json(obj.detach().cpu().tolist())
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k) if not isinstance(k, str) else k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return [_sanitize_for_json(x) for x in obj]
    return obj


def _run_coroutine_sync(coro):
    """Run an async coroutine synchronously from sync code, handling existing event loops safely."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If already running inside an asyncio event loop thread, execute in a worker thread
        result = None
        error = None

        def _worker():
            nonlocal result, error
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(coro)
            except Exception as e:
                error = e
            finally:
                new_loop.close()

        thread = threading.Thread(target=_worker)
        thread.start()
        thread.join()
        if error:
            raise error
        return result
    else:
        return asyncio.run(coro)


class TrainPilotMCPClient:
    """Client for calling TrainPilot MCP Server from GPU nodes."""

    def __init__(
        self,
        server_url: Optional[str] = None,
        task_id: Optional[str] = None,
        timeout: float = 30.0,
        api_token: Optional[str] = None,
    ):
        """Initialize MCP client.

        Args:
            server_url: Base URL of the TrainPilot server (e.g. 'http://公网IP:28780'
                or 'http://公网IP:28780/sse'). If omitted, resolves from environment variables
                TRAINPILOT_GATEWAY_URL or TRAINPILOT_HOST:TRAINPILOT_PORT.
            task_id: Default task identifier. Resolves from TRAINPILOT_TASK_ID if omitted.
            timeout: Default timeout in seconds for operations.
            api_token: Optional API token for Bearer authorization. Resolves from TRAINPILOT_API_TOKEN if omitted.
        """
        raw_url = server_url or os.getenv("TRAINPILOT_GATEWAY_URL")
        if not raw_url:
            host = os.getenv("TRAINPILOT_HOST", "127.0.0.1")
            port = os.getenv("TRAINPILOT_PORT", "28780")
            raw_url = f"http://{host}:{port}"

        raw_url = raw_url.rstrip("/")
        if raw_url.endswith("/sse"):
            raw_url = raw_url[:-4]
        if raw_url.endswith("/mcp"):
            self.mcp_url = raw_url
            self.base_url = raw_url[:-4]
        else:
            self.mcp_url = f"{raw_url}/mcp"
            self.base_url = raw_url

        # Backwards compatibility alias
        self.sse_url = self.mcp_url
        self.default_task_id = task_id or os.getenv("TRAINPILOT_TASK_ID", "default-task")
        self.timeout = timeout
        self.api_token = api_token or os.getenv("TRAINPILOT_API_TOKEN")

    def _resolve_task_id(self, task_id: Optional[str]) -> str:
        tid = task_id or self.default_task_id
        if not tid:
            raise ValueError("task_id must be provided or configured via TRAINPILOT_TASK_ID")
        return tid

    def _create_http_client(self):
        """Create an AsyncClient configured with timeouts and optional API token headers."""
        try:
            import httpx2 as _http_mod
        except ImportError:
            import httpx as _http_mod

        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return _http_mod.AsyncClient(headers=headers, timeout=self.timeout)

    async def call_tool_async(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the remote TrainPilot MCP Server asynchronously over Streamable HTTP."""
        sanitized_args = _sanitize_for_json(arguments)
        async with self._create_http_client() as http_client:
            async with streamable_http_client(self.mcp_url, http_client=http_client) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    response = await session.call_tool(tool_name, sanitized_args)
                    if response.content:
                        first = response.content[0]
                        if hasattr(first, "text"):
                            try:
                                return json.loads(first.text)
                            except Exception:
                                return first.text
                    return response

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Synchronously call a tool on the remote TrainPilot MCP Server."""
        return _run_coroutine_sync(self.call_tool_async(tool_name, arguments))

    async def list_tools_async(self) -> List[str]:
        """List all available tools on the remote MCP Server asynchronously over Streamable HTTP."""
        async with self._create_http_client() as http_client:
            async with streamable_http_client(self.mcp_url, http_client=http_client) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_res = await session.list_tools()
                    return [t.name for t in tools_res.tools]

    def list_tools(self) -> List[str]:
        """Synchronously list all available tools on the remote MCP Server."""
        return _run_coroutine_sync(self.list_tools_async())

    # High-level tool wrapper methods

    def report_milestone(
        self,
        message: str,
        task_id: Optional[str] = None,
        step: int = 0,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        agent_note: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        fail_silently: bool = True,
    ) -> Dict[str, Any]:
        """Report training milestone with AI note via MCP Server."""
        tid = self._resolve_task_id(task_id)
        args: Dict[str, Any] = {
            "task_id": tid,
            "message": message,
            "step": step,
        }
        if epoch is not None:
            args["epoch"] = epoch
        if metrics is not None:
            args["metrics"] = metrics
        if agent_note is not None:
            args["agent_note"] = agent_note
        if extra is not None:
            args["extra"] = extra

        try:
            return self.call_tool("report_milestone", args)
        except Exception as exc:
            if fail_silently:
                logger.warning("[%s] Failed to send milestone via MCP: %s (continuing training)", tid, exc)
                return {"success": False, "task_id": tid, "error": str(exc), "fail_silently": True}
            raise

    def report_alert(
        self,
        message: str,
        task_id: Optional[str] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        agent_note: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Report training anomaly and freeze training via MCP Server."""
        tid = self._resolve_task_id(task_id)
        args: Dict[str, Any] = {
            "task_id": tid,
            "message": message,
        }
        if step is not None:
            args["step"] = step
        if epoch is not None:
            args["epoch"] = epoch
        if metrics is not None:
            args["metrics"] = metrics
        if agent_note is not None:
            args["agent_note"] = agent_note
        if extra is not None:
            args["extra"] = extra

        return self.call_tool("report_alert", args)

    def poll_instruction(
        self,
        task_id: Optional[str] = None,
        wait_timeout: float = 20.0,
        pop: bool = True,
        timeout: Optional[float] = None,
        interval: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Poll or long-poll for human instructions via MCP Server.

        If timeout is provided and > 0, loops until an instruction becomes ready or timeout occurs.
        """
        tid = self._resolve_task_id(task_id)
        if timeout is not None and timeout > 0:
            start_time = time.time()
            while True:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(f"Timed out after {timeout} seconds waiting for decision on task {tid}")
                current_wait = max(0.5, min(wait_timeout, timeout - elapsed))
                res = self.call_tool("poll_instruction", {
                    "task_id": tid,
                    "wait_timeout": current_wait,
                    "pop": pop,
                })
                if isinstance(res, dict) and (res.get("ready") or res.get("has_instruction")):
                    return res
                time.sleep(interval)
        else:
            return self.call_tool("poll_instruction", {
                "task_id": tid,
                "wait_timeout": wait_timeout,
                "pop": pop,
            })

    def ack_instruction(
        self,
        action: str = "self_resolve",
        status: str = "success",
        solution: str = "",
        message: str = "",
        task_id: Optional[str] = None,
        instruction_id: Optional[str] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Acknowledge instruction execution via MCP Server."""
        tid = self._resolve_task_id(task_id)
        args: Dict[str, Any] = {
            "task_id": tid,
            "action": action,
            "status": status,
            "solution": solution,
            "message": message,
        }
        if instruction_id is not None:
            args["instruction_id"] = instruction_id
        if step is not None:
            args["step"] = step
        if epoch is not None:
            args["epoch"] = epoch
        if metrics is not None:
            args["metrics"] = metrics

        return self.call_tool("ack_instruction", args)

    def send_heartbeat(
        self,
        task_id: Optional[str] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        status: str = "running",
    ) -> Dict[str, Any]:
        """Send heartbeat telemetry via MCP Server."""
        tid = self._resolve_task_id(task_id)
        args: Dict[str, Any] = {
            "task_id": tid,
            "status": status,
        }
        if step is not None:
            args["step"] = step
        if epoch is not None:
            args["epoch"] = epoch
        if metrics is not None:
            args["metrics"] = metrics

        try:
            return self.call_tool("send_heartbeat", args)
        except Exception as exc:
            logger.warning("[%s] Failed to send heartbeat via MCP: %s", tid, exc)
            return {"success": False, "task_id": tid, "error": str(exc)}

    def get_task_status(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Query task state and mailbox via MCP Server."""
        tid = self._resolve_task_id(task_id)
        return self.call_tool("get_task_status", {"task_id": tid})

    def list_tasks(
        self,
        limit: int = 100,
        offset: int = 0,
        state: Optional[str] = None,
        stale_only: bool = False,
    ) -> Dict[str, Any]:
        """List tasks via MCP Server."""
        args: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "stale_only": stale_only,
        }
        if state is not None:
            args["state"] = state
        return self.call_tool("list_tasks", args)

    def submit_decision(
        self,
        action: str,
        task_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        operator: str = "mcp_client",
    ) -> Dict[str, Any]:
        """Submit a decision directly via MCP Server."""
        tid = self._resolve_task_id(task_id)
        args: Dict[str, Any] = {
            "task_id": tid,
            "action": action,
            "operator": operator,
        }
        if payload is not None:
            args["payload"] = payload
        return self.call_tool("submit_decision", args)

    # Duck-typing aliases for seamless TrainingGuardian and TrainPilotClient compatibility
    notify_milestone = report_milestone
    notify_alert = report_alert
    heartbeat = send_heartbeat
    get_status = get_task_status
