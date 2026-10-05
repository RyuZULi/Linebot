"""
持久化紀錄層：每次熱量估算/使用者回報都存進 SQLite，附時間戳記，
支援依時間範圍刪除。取代原本純記憶體的 pending_meals（服務重啟不會
再遺失歷史紀錄）。
"""

import json
import sqlite3
from datetime import datetime, timedelta

DB_PATH = r"D:\CalorieCalculation\data\line_bot.db"

TIME_WINDOWS = {
    "morning": (0, 12),
    "afternoon": (12, 18),
    "evening": (18, 24),
}


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meal_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,        -- 'estimated' 或 'user_reported'
            summary TEXT NOT NULL,       -- 一行總結，列紀錄時顯示
            detail TEXT NOT NULL,        -- 完整內容，"細節"指令時顯示
            total_calories REAL,
            raw_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_record(user_id: str, source: str, summary: str, detail: str, total_calories, raw: dict = None) -> int:
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO meal_records (user_id, created_at, source, summary, detail, total_calories, raw_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            user_id,
            datetime.now().isoformat(timespec="seconds"),
            source,
            summary,
            detail,
            total_calories,
            json.dumps(raw, ensure_ascii=False) if raw else None,
        ),
    )
    conn.commit()
    record_id = cur.lastrowid
    conn.close()
    return record_id


def get_recent(user_id: str, limit: int = 10) -> list:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM meal_records WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest(user_id: str) -> dict:
    rows = get_recent(user_id, limit=1)
    return rows[0] if rows else None


def daily_totals(user_id: str, start_date, end_date) -> list:
    """回傳 [{"date": "YYYY-MM-DD", "total": float, "count": int}, ...]，
    只包含期間內有紀錄的日子（start_date、end_date 都含，型別是 date）。"""
    conn = _connect()
    rows = conn.execute(
        "SELECT substr(created_at, 1, 10) AS day, SUM(total_calories) AS total, COUNT(*) AS cnt "
        "FROM meal_records WHERE user_id = ? AND total_calories IS NOT NULL "
        "AND substr(created_at, 1, 10) BETWEEN ? AND ? GROUP BY day ORDER BY day",
        (user_id, start_date.isoformat(), end_date.isoformat()),
    ).fetchall()
    conn.close()
    return [{"date": r["day"], "total": round(r["total"], 1), "count": r["cnt"]} for r in rows]


def delete_records(user_id: str, date_filter: str = "today", time_of_day: str = None) -> int:
    """date_filter: "today" 或 "yesterday"；time_of_day: "morning"/"afternoon"/"evening"/None(整天)。
    回傳刪除的筆數。"""
    today = datetime.now().date()
    target_date = today if date_filter == "today" else today - timedelta(days=1)

    if time_of_day and time_of_day in TIME_WINDOWS:
        start_h, end_h = TIME_WINDOWS[time_of_day]
    else:
        start_h, end_h = 0, 24

    start = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=start_h)
    end = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=end_h)

    conn = _connect()
    cur = conn.execute(
        "DELETE FROM meal_records WHERE user_id = ? AND created_at >= ? AND created_at < ?",
        (user_id, start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")),
    )
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return deleted


if __name__ == "__main__":
    init_db()
    print(f"initialized {DB_PATH}")
