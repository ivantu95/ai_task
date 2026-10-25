"""
Database layer. Uses plain sqlite3 (no ORM) since this is a small personal
app - a single file database is enough, no separate DB server needed.
"""
import sqlite3
from datetime import datetime, date
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "tasks.db"


def init_db():
    """Create tables if they don't exist yet. Call once at startup."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            chat_id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL DEFAULT 0,
            text TEXT NOT NULL,
            due_time TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            priority TEXT NOT NULL DEFAULT 'normal',
            snooze_count INTEGER NOT NULL DEFAULT 0,
            is_recurring INTEGER NOT NULL DEFAULT 0,
            recurring_time TEXT,
            exclude_days TEXT,
            completed_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recurring_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            chat_id INTEGER NOT NULL,
            log_date TEXT NOT NULL,
            completed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    # Migrations for columns added after the table was first created.
    existing_columns = [row[1] for row in conn.execute("PRAGMA table_info(tasks)")]
    if "chat_id" not in existing_columns:
        conn.execute("ALTER TABLE tasks ADD COLUMN chat_id INTEGER NOT NULL DEFAULT 0")
    if "is_recurring" not in existing_columns:
        conn.execute("ALTER TABLE tasks ADD COLUMN is_recurring INTEGER NOT NULL DEFAULT 0")
    if "recurring_time" not in existing_columns:
        conn.execute("ALTER TABLE tasks ADD COLUMN recurring_time TEXT")
    if "exclude_days" not in existing_columns:
        conn.execute("ALTER TABLE tasks ADD COLUMN exclude_days TEXT")
    if "priority" not in existing_columns:
        conn.execute("ALTER TABLE tasks ADD COLUMN priority TEXT NOT NULL DEFAULT 'normal'")
    if "completed_at" not in existing_columns:
        conn.execute("ALTER TABLE tasks ADD COLUMN completed_at TEXT")
    if "snooze_count" not in existing_columns:
        conn.execute("ALTER TABLE tasks ADD COLUMN snooze_count INTEGER NOT NULL DEFAULT 0")

    conn.commit()
    conn.close()


def register_user(chat_id: int):
    """Records that this chat_id has talked to the bot, so the daily
    reminder job knows who to message."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR IGNORE INTO users (chat_id, created_at) VALUES (?, ?)",
        (chat_id, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def list_all_chat_ids() -> list[int]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT chat_id FROM users").fetchall()
    conn.close()
    return [row[0] for row in rows]


def add_task(
    chat_id: int, text: str, due_time: str | None = None, priority: str = "normal", snooze_count: int = 0
) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        """INSERT INTO tasks (chat_id, text, due_time, status, created_at, priority, snooze_count)
           VALUES (?, ?, ?, 'pending', ?, ?, ?)""",
        (chat_id, text, due_time, datetime.now().isoformat(), priority, snooze_count),
    )
    conn.commit()
    task_id = cur.lastrowid
    conn.close()
    return task_id


def list_pending_tasks(chat_id: int) -> list[dict]:
    """Only returns one-time tasks - recurring reminders are managed
    separately since they never move to 'done', they just repeat.
    High priority tasks are sorted to the top."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id, text, due_time, priority, snooze_count FROM tasks
           WHERE chat_id = ? AND status = 'pending' AND is_recurring = 0
           ORDER BY CASE WHEN priority = 'high' THEN 0 ELSE 1 END, id""",
        (chat_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def mark_done(chat_id: int, task_id: int) -> bool:
    """Returns True if a pending task with this id, belonging to this
    chat_id, was found and marked done. The chat_id check stops one user
    from marking another user's task as done."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ? AND chat_id = ? AND status = 'pending'",
        (datetime.now().isoformat(), task_id, chat_id),
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def add_recurring_task(chat_id: int, text: str, recurring_time: str, exclude_days: list[str]) -> int:
    """exclude_days is a list like ['sun'] - days to SKIP the reminder."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        """INSERT INTO tasks (chat_id, text, status, created_at, is_recurring, recurring_time, exclude_days)
           VALUES (?, ?, 'pending', ?, 1, ?, ?)""",
        (chat_id, text, datetime.now().isoformat(), recurring_time, ",".join(exclude_days)),
    )
    conn.commit()
    task_id = cur.lastrowid
    conn.close()
    return task_id


def list_recurring_tasks(chat_id: int) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, text, recurring_time, exclude_days FROM tasks WHERE chat_id = ? AND is_recurring = 1 ORDER BY id",
        (chat_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_task(chat_id: int, task_id: int) -> bool:
    """Hard delete - used both to auto-clean a one-time reminder after it
    fires, and to stop a recurring reminder permanently."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("DELETE FROM tasks WHERE id = ? AND chat_id = ?", (task_id, chat_id))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted




def get_weekly_stats(chat_id: int) -> dict:
    """Compares this week vs last week - created tasks, completed tasks,
    and how high-priority tasks specifically were handled."""
    from datetime import timedelta

    conn = sqlite3.connect(DB_PATH)
    now = datetime.now()
    week_start = (now - timedelta(days=7)).isoformat()
    prev_week_start = (now - timedelta(days=14)).isoformat()

    def count(query, params):
        return conn.execute(query, params).fetchone()[0]

    created_this_week = count(
        "SELECT COUNT(*) FROM tasks WHERE chat_id=? AND is_recurring=0 AND created_at >= ?",
        (chat_id, week_start),
    )
    done_this_week = count(
        "SELECT COUNT(*) FROM tasks WHERE chat_id=? AND is_recurring=0 AND status='done' AND completed_at >= ?",
        (chat_id, week_start),
    )
    created_prev_week = count(
        "SELECT COUNT(*) FROM tasks WHERE chat_id=? AND is_recurring=0 AND created_at >= ? AND created_at < ?",
        (chat_id, prev_week_start, week_start),
    )
    done_prev_week = count(
        "SELECT COUNT(*) FROM tasks WHERE chat_id=? AND is_recurring=0 AND status='done' AND completed_at >= ? AND completed_at < ?",
        (chat_id, prev_week_start, week_start),
    )
    high_priority_created = count(
        "SELECT COUNT(*) FROM tasks WHERE chat_id=? AND is_recurring=0 AND priority='high' AND created_at >= ?",
        (chat_id, week_start),
    )
    high_priority_done = count(
        "SELECT COUNT(*) FROM tasks WHERE chat_id=? AND is_recurring=0 AND priority='high' AND status='done' AND completed_at >= ?",
        (chat_id, week_start),
    )
    conn.close()

    return {
        "created_this_week": created_this_week,
        "done_this_week": done_this_week,
        "completion_rate": round(done_this_week / created_this_week * 100) if created_this_week else 0,
        "prev_completion_rate": round(done_prev_week / created_prev_week * 100) if created_prev_week else 0,
        "high_priority_created": high_priority_created,
        "high_priority_done": high_priority_done,
    }


def log_recurring_fired(chat_id: int, task_id: int):
    """Called every time a recurring reminder actually fires - records
    that today's occurrence happened, so we can later check if it was
    completed or not."""
    conn = sqlite3.connect(DB_PATH)
    today = date.today().isoformat()
    conn.execute(
        "INSERT INTO recurring_logs (task_id, chat_id, log_date, completed, created_at) VALUES (?, ?, ?, 0, ?)",
        (task_id, chat_id, today, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def mark_recurring_done(chat_id: int, task_id: int):
    """Marks today's occurrence of a recurring task as completed."""
    conn = sqlite3.connect(DB_PATH)
    today = date.today().isoformat()
    cur = conn.execute(
        "UPDATE recurring_logs SET completed = 1 WHERE task_id = ? AND chat_id = ? AND log_date = ? AND completed = 0",
        (task_id, chat_id, today),
    )
    if cur.rowcount == 0:
        # No log row yet for today (e.g. user marked it done before the
        # scheduled reminder fired) - just insert a completed row directly.
        conn.execute(
            "INSERT INTO recurring_logs (task_id, chat_id, log_date, completed, created_at) VALUES (?, ?, ?, 1, ?)",
            (task_id, chat_id, today, datetime.now().isoformat()),
        )
    conn.commit()
    conn.close()


def get_recurring_weekly_stats(chat_id: int) -> dict:
    from datetime import timedelta
    conn = sqlite3.connect(DB_PATH)
    week_start = (date.today() - timedelta(days=7)).isoformat()

    fired = conn.execute(
        "SELECT COUNT(*) FROM recurring_logs WHERE chat_id = ? AND log_date >= ?",
        (chat_id, week_start),
    ).fetchone()[0]
    completed = conn.execute(
        "SELECT COUNT(*) FROM recurring_logs WHERE chat_id = ? AND log_date >= ? AND completed = 1",
        (chat_id, week_start),
    ).fetchone()[0]
    conn.close()

    return {
        "recurring_fired_this_week": fired,
        "recurring_completed_this_week": completed,
        "recurring_completion_rate": round(completed / fired * 100) if fired else 0,
    }


def search_tasks(chat_id: int, query: str) -> list[dict]:
    """Simple substring search over pending one-time tasks."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id, text, due_time, priority FROM tasks
           WHERE chat_id = ? AND status = 'pending' AND is_recurring = 0
           AND text LIKE ?
           ORDER BY CASE WHEN priority = 'high' THEN 0 ELSE 1 END, id""",
        (chat_id, f"%{query}%"),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]



def archive_fired_task(chat_id: int, task_id: int):
    """Called when a one-time reminder fires, INSTEAD of deleting it. Keeps
    the row around (status='fired') so it can be found later via history
    search - we just don't know for sure if the user actually did it."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE tasks SET status = 'fired', completed_at = ? WHERE id = ? AND chat_id = ?",
        (datetime.now().isoformat(), task_id, chat_id),
    )
    conn.commit()
    conn.close()


def get_task_history(chat_id: int, limit: int = 200) -> list[dict]:
    """Returns past tasks - both explicitly completed ('done') and reminders
    that fired ('fired') - most recent first, for semantic history search."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id, text, status, priority, created_at, completed_at FROM tasks
           WHERE chat_id = ? AND status IN ('done', 'fired')
           ORDER BY COALESCE(completed_at, created_at) DESC LIMIT ?""",
        (chat_id, limit),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]