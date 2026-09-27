"""
開發任務的持久化紀錄層：每個「任務：...」指令建立一筆紀錄，追蹤它
從 pending（待處理）→ running（Claude 在隔離分支裡跑）→
awaiting_confirmation（跑完、健檢通過，等主人確認要不要套用）→
merged/rejected/failed 的狀態，方便重啟服務後還查得到進行中的任務。
"""

import sqlite3
from datetime import datetime

DB_PATH = r"D:\CalorieCalculation\data\dev_tasks.db"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dev_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            requester_user_id TEXT NOT NULL,
            description TEXT NOT NULL,
            status TEXT NOT NULL,        -- pending/running/awaiting_confirmation/merged/rejected/failed
            branch_name TEXT,
            worktree_path TEXT,
            result_summary TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def create_task(requester_user_id: str, description: str) -> int:
    conn = _connect()
    now = datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO dev_tasks (created_at, updated_at, requester_user_id, description, status) "
        "VALUES (?, ?, ?, ?, 'pending')",
        (now, now, requester_user_id, description),
    )
    conn.commit()
    task_id = cur.lastrowid
    conn.close()
    return task_id


def update_task(task_id: int, **fields) -> None:
    """fields 可以是 status/branch_name/worktree_path/result_summary 任意組合。"""
    if not fields:
        return
    fields["updated_at"] = datetime.now().isoformat(timespec="seconds")
    columns = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [task_id]
    conn = _connect()
    conn.execute(f"UPDATE dev_tasks SET {columns} WHERE id = ?", values)
    conn.commit()
    conn.close()


def get_task(task_id: int) -> dict:
    conn = _connect()
    row = conn.execute("SELECT * FROM dev_tasks WHERE id = ?", (task_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_recent(limit: int = 10) -> list:
    conn = _connect()
    rows = conn.execute("SELECT * FROM dev_tasks ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"initialized {DB_PATH}")
