"""SQLite local persistence for TrainPilot tasks and event histories."""

import json
import logging
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from trainpilot.common.schemas import InstructionResponse
from trainpilot.common.states import TaskState

logger = logging.getLogger("trainpilot.storage")


class SQLiteStorage:
    """Thread-safe SQLite persistent storage for TrainPilot.

    Features:
    - WAL journal mode for high concurrency.
    - Automatic retention trimming: when task events exceed max_events,
      oldest records are automatically pruned.
    - Full task state restoration on server restart.
    """

    def __init__(self, db_path: str = "trainpilot.db"):
        self.db_path = db_path
        # Ensure directory exists if path contains directories
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Initialize tables and pragmas."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA auto_vacuum=INCREMENTAL;")
            cur.execute("PRAGMA wal_autocheckpoint=1000;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_heartbeat_at TEXT,
                    latest_step INTEGER,
                    latest_epoch INTEGER,
                    latest_metrics_json TEXT,
                    latest_message TEXT,
                    pending_instruction_json TEXT,
                    latest_instruction_json TEXT,
                    last_alert_fingerprint TEXT,
                    last_alert_at TEXT,
                    stale_alerted INTEGER DEFAULT 0,
                    feishu_message_id TEXT
                );
                """
            )
            # Automatic column migration if table already existed without feishu_message_id
            cur.execute("PRAGMA table_info(tasks);")
            cols = [r["name"] for r in cur.fetchall()]
            if "feishu_message_id" not in cols:
                cur.execute("ALTER TABLE tasks ADD COLUMN feishu_message_id TEXT;")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT,
                    step INTEGER,
                    epoch INTEGER,
                    metrics_json TEXT,
                    extra_json TEXT,
                    agent_note TEXT,
                    timestamp TEXT NOT NULL
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_task_id_id ON events(task_id, id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_updated_at ON tasks(updated_at);"
            )
            self._conn.commit()
            logger.info("SQLite storage initialized at: %s", self.db_path)

    def save_task(self, task: Any, stale_alerted: Optional[bool] = None) -> None:
        """Upsert a task record into SQLite."""
        pending_json = (
            task.pending_instruction.model_dump_json()
            if getattr(task, "pending_instruction", None)
            else None
        )
        latest_metrics_json = (
            json.dumps(task.latest_metrics)
            if getattr(task, "latest_metrics", None)
            else None
        )
        latest_inst_json = (
            json.dumps(task.latest_instruction)
            if getattr(task, "latest_instruction", None)
            else None
        )
        feishu_msg_id = getattr(task, "feishu_message_id", None)

        with self._lock:
            cur = self._conn.cursor()
            if stale_alerted is not None:
                alerted_val = 1 if stale_alerted else 0
            else:
                # Check existing stale_alerted and feishu_message_id if already exists
                cur.execute(
                    "SELECT stale_alerted, feishu_message_id FROM tasks WHERE task_id = ?",
                    (task.task_id,),
                )
                row = cur.fetchone()
                alerted_val = row["stale_alerted"] if row else 0
                if feishu_msg_id is None and row:
                    feishu_msg_id = row["feishu_message_id"]

            cur.execute(
                """
                INSERT INTO tasks (
                    task_id, state, created_at, updated_at, last_heartbeat_at,
                    latest_step, latest_epoch, latest_metrics_json, latest_message,
                    pending_instruction_json, latest_instruction_json,
                    last_alert_fingerprint, last_alert_at, stale_alerted, feishu_message_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    state = excluded.state,
                    updated_at = excluded.updated_at,
                    last_heartbeat_at = excluded.last_heartbeat_at,
                    latest_step = excluded.latest_step,
                    latest_epoch = excluded.latest_epoch,
                    latest_metrics_json = excluded.latest_metrics_json,
                    latest_message = excluded.latest_message,
                    pending_instruction_json = excluded.pending_instruction_json,
                    latest_instruction_json = excluded.latest_instruction_json,
                    last_alert_fingerprint = excluded.last_alert_fingerprint,
                    last_alert_at = excluded.last_alert_at,
                    stale_alerted = excluded.stale_alerted,
                    feishu_message_id = COALESCE(excluded.feishu_message_id, tasks.feishu_message_id);
                """,
                (
                    task.task_id,
                    task.state.value if hasattr(task.state, "value") else str(task.state),
                    task.created_at,
                    task.updated_at,
                    task.last_heartbeat_at,
                    task.latest_step,
                    task.latest_epoch,
                    latest_metrics_json,
                    task.latest_message,
                    pending_json,
                    latest_inst_json,
                    task.last_alert_fingerprint,
                    task.last_alert_at,
                    alerted_val,
                    feishu_msg_id,
                ),
            )
            self._conn.commit()

    def set_task_feishu_message_id(self, task_id: str, message_id: str) -> None:
        """Store the Feishu message_id associated with this task (survives restarts)."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE tasks SET feishu_message_id = ? WHERE task_id = ?",
                (message_id, task_id),
            )
            self._conn.commit()

    def get_task_feishu_message_id(self, task_id: str) -> Optional[str]:
        """Fetch the Feishu message_id for updating interactive cards."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT feishu_message_id FROM tasks WHERE task_id = ?",
                (task_id,),
            )
            row = cur.fetchone()
            return row["feishu_message_id"] if row else None

    def set_task_stale_alerted(self, task_id: str, alerted: bool) -> None:
        """Mark whether watchdog has dispatched a stale alert for task_id (debounce)."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE tasks SET stale_alerted = ? WHERE task_id = ?",
                (1 if alerted else 0, task_id),
            )
            self._conn.commit()

    def is_task_stale_alerted(self, task_id: str) -> bool:
        """Check if watchdog has already sent a stale alert."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT stale_alerted FROM tasks WHERE task_id = ?", (task_id,)
            )
            row = cur.fetchone()
            if row:
                return bool(row["stale_alerted"])
            return False

    def append_event(
        self, task_id: str, event_entry: Dict[str, Any], max_events: int = 500
    ) -> None:
        """Append an event to events table and prune old events if exceeding max_events."""
        metrics_json = (
            json.dumps(event_entry.get("metrics"))
            if event_entry.get("metrics") is not None
            else None
        )
        extra_json = (
            json.dumps(event_entry.get("extra"))
            if event_entry.get("extra") is not None
            else None
        )

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO events (
                    task_id, event_type, message, step, epoch,
                    metrics_json, extra_json, agent_note, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    event_entry.get("event_type"),
                    event_entry.get("message"),
                    event_entry.get("step"),
                    event_entry.get("epoch"),
                    metrics_json,
                    extra_json,
                    event_entry.get("agent_note"),
                    event_entry.get("timestamp"),
                ),
            )

            # Auto retention trimming: prune oldest events if count > max_events
            if max_events > 0:
                cur.execute(
                    """
                    DELETE FROM events
                    WHERE task_id = ?
                      AND id NOT IN (
                        SELECT id FROM events
                        WHERE task_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                      )
                    """,
                    (task_id, task_id, max_events),
                )

            self._conn.commit()

    def get_events(
        self, task_id: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Fetch paginated events in chronological order."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT event_type, message, step, epoch, metrics_json, extra_json, agent_note, timestamp
                FROM events
                WHERE task_id = ?
                ORDER BY id ASC
                LIMIT ? OFFSET ?
                """,
                (task_id, limit, offset),
            )
            rows = cur.fetchall()

            events = []
            for r in rows:
                metrics = json.loads(r["metrics_json"]) if r["metrics_json"] else None
                extra = json.loads(r["extra_json"]) if r["extra_json"] else None
                events.append(
                    {
                        "event_type": r["event_type"],
                        "message": r["message"],
                        "step": r["step"],
                        "epoch": r["epoch"],
                        "metrics": metrics,
                        "extra": extra,
                        "agent_note": r["agent_note"],
                        "timestamp": r["timestamp"],
                    }
                )
            return events

    def get_events_count(self, task_id: str) -> int:
        """Count total events stored for task_id."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT COUNT(*) as cnt FROM events WHERE task_id = ?", (task_id,)
            )
            row = cur.fetchone()
            return row["cnt"] if row else 0

    def trim_all_events(self, max_events: int = 500) -> int:
        """Prune excess events across all tasks (periodic maintenance)."""
        with self._lock:
            cur = self._conn.cursor()
            # Get distinct tasks that have excess events
            cur.execute(
                """
                SELECT task_id, COUNT(*) as cnt
                FROM events
                GROUP BY task_id
                HAVING cnt > ?
                """,
                (max_events,),
            )
            tasks_to_trim = cur.fetchall()
            trimmed_count = 0
            for t in tasks_to_trim:
                t_id = t["task_id"]
                cur.execute(
                    """
                    DELETE FROM events
                    WHERE task_id = ?
                      AND id NOT IN (
                        SELECT id FROM events
                        WHERE task_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                      )
                    """,
                    (t_id, t_id, max_events),
                )
                trimmed_count += cur.rowcount
            self._conn.commit()
            return trimmed_count

    def _row_to_task_record(self, r: Any) -> Any:
        """Convert a sqlite3.Row to a TaskRecord."""
        from trainpilot.server.mailbox import TaskRecord

        t_id = r["task_id"]
        pending_inst = None
        if r["pending_instruction_json"]:
            try:
                pending_inst = InstructionResponse.model_validate_json(
                    r["pending_instruction_json"]
                )
            except Exception as e:
                logger.warning(
                    "Failed to parse pending_instruction for %s: %s", t_id, e
                )

        latest_inst = None
        if r["latest_instruction_json"]:
            try:
                latest_inst = json.loads(r["latest_instruction_json"])
            except Exception as e:
                logger.warning(
                    "Failed to parse latest_instruction for %s: %s", t_id, e
                )

        latest_metrics = None
        if r["latest_metrics_json"]:
            try:
                latest_metrics = json.loads(r["latest_metrics_json"])
            except Exception as e:
                logger.warning(
                    "Failed to parse latest_metrics for %s: %s", t_id, e
                )

        keys = r.keys()
        feishu_msg_id = r["feishu_message_id"] if "feishu_message_id" in keys else None

        return TaskRecord(
            task_id=t_id,
            state=TaskState(r["state"]),
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            last_heartbeat_at=r["last_heartbeat_at"],
            latest_step=r["latest_step"],
            latest_epoch=r["latest_epoch"],
            latest_metrics=latest_metrics,
            latest_message=r["latest_message"],
            pending_instruction=pending_inst,
            latest_instruction=latest_inst,
            last_alert_fingerprint=r["last_alert_fingerprint"],
            last_alert_at=r["last_alert_at"],
            feishu_message_id=feishu_msg_id,
        )

    def get_task(self, task_id: str) -> Optional[Any]:
        """Fetch a single task record directly from SQLite."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_task_record(row)

    def list_tasks(
        self,
        limit: int = 100,
        offset: int = 0,
        state: Optional[TaskState] = None,
    ) -> List[Any]:
        """Fetch paginated task records from SQLite."""
        with self._lock:
            cur = self._conn.cursor()
            if state is not None:
                st_val = state.value if hasattr(state, "value") else str(state)
                cur.execute(
                    "SELECT * FROM tasks WHERE state = ? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                    (st_val, limit, offset),
                )
            else:
                cur.execute(
                    "SELECT * FROM tasks ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                )
            rows = cur.fetchall()
            return [self._row_to_task_record(r) for r in rows]

    def get_tasks_count(self, state: Optional[TaskState] = None) -> int:
        """Fast SQL COUNT(*) without materializing task objects in memory."""
        with self._lock:
            cur = self._conn.cursor()
            if state is not None:
                st_val = state.value if hasattr(state, "value") else str(state)
                cur.execute("SELECT COUNT(*) as cnt FROM tasks WHERE state = ?", (st_val,))
            else:
                cur.execute("SELECT COUNT(*) as cnt FROM tasks")
            row = cur.fetchone()
            return row["cnt"] if row else 0

    def checkpoint_and_vacuum(self) -> Dict[str, Any]:
        """Perform WAL checkpoint and incremental vacuum to reclaim disk space."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            ckpt_res = cur.fetchone()
            cur.execute("PRAGMA incremental_vacuum;")
            self._conn.commit()
            return {
                "checkpoint": list(ckpt_res) if ckpt_res else None,
            }

    def clean_expired_tasks(self, max_age_seconds: int = 86400 * 7) -> int:
        """Clean up old finished tasks (COMPLETED/FAILED) older than max_age_seconds to free disk space."""
        from datetime import datetime, timezone, timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)).isoformat()
        with self._lock:
            cur = self._conn.cursor()
            terminal_states = (TaskState.COMPLETED.value, TaskState.FAILED.value)
            cur.execute(
                """
                SELECT task_id FROM tasks
                WHERE state IN (?, ?) AND updated_at < ?
                """,
                (terminal_states[0], terminal_states[1], cutoff),
            )
            old_tasks = [r["task_id"] for r in cur.fetchall()]
            if not old_tasks:
                return 0
            for tid in old_tasks:
                cur.execute("DELETE FROM events WHERE task_id = ?", (tid,))
                cur.execute("DELETE FROM tasks WHERE task_id = ?", (tid,))
            self._conn.commit()
            logger.info("Cleaned %d expired finished tasks older than %ds", len(old_tasks), max_age_seconds)
            return len(old_tasks)

    def load_all_tasks(self, active_only: bool = False) -> Dict[str, Any]:
        """Load task records from SQLite. If active_only=True, only non-terminal tasks are loaded."""
        with self._lock:
            cur = self._conn.cursor()
            if active_only:
                terminal_states = (TaskState.COMPLETED.value, TaskState.FAILED.value)
                cur.execute("SELECT * FROM tasks WHERE state NOT IN (?, ?)", terminal_states)
            else:
                cur.execute("SELECT * FROM tasks")
            rows = cur.fetchall()
            tasks: Dict[str, Any] = {}
            for r in rows:
                record = self._row_to_task_record(r)
                tasks[record.task_id] = record
            return tasks

    def reset(self) -> None:
        """Clear all tasks and events (mainly for tests)."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM events;")
            cur.execute("DELETE FROM tasks;")
            self._conn.commit()

    def close(self) -> None:
        """Close SQLite connection."""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
