#############################################################################
# weBIGeo Clouds
# Copyright (C) 2026 Wendelin Muth
# Copyright (C) 2026 Gerald Kimmersdorfer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#############################################################################

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

_conn: sqlite3.Connection | None = None
_lock = threading.Lock()


def init(path: str = "data/clouds-server.db") -> None:
    """Open (or create) the SQLite database and initialise all tables."""
    global _conn
    _conn = sqlite3.connect(path, check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    with _lock:
        # Migrate old table name if present
        try:
            _conn.execute("ALTER TABLE tile_cache RENAME TO tilesets")
            _conn.commit()
        except sqlite3.OperationalError:
            pass  # already renamed or never existed

        _conn.executescript("""
            CREATE TABLE IF NOT EXISTS public_log (
                id  INTEGER PRIMARY KEY AUTOINCREMENT,
                dt  TEXT NOT NULL,
                msg TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_public_log_dt ON public_log(dt);

            CREATE TABLE IF NOT EXISTS maintenance (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                label        TEXT NOT NULL,
                started_at   TEXT NOT NULL,
                completed_at TEXT,
                added        TEXT NOT NULL,
                purged       TEXT NOT NULL,
                renewed      TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS tilesets (
                folder         TEXT PRIMARY KEY,
                run_str        TEXT NOT NULL,
                step           INTEGER NOT NULL,
                target_str     TEXT NOT NULL,
                status         TEXT NOT NULL,
                size           INTEGER,
                queued_at      TEXT NOT NULL,
                completed_at   TEXT,
                maintenance_id INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_tilesets_status ON tilesets(status);
            CREATE INDEX IF NOT EXISTS idx_tilesets_target ON tilesets(target_str);
        """)
        _conn.commit()

        # Idempotent migration: add maintenance_id if missing (pre-rename databases)
        try:
            _conn.execute("ALTER TABLE tilesets ADD COLUMN maintenance_id INTEGER")
            _conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists


# ---------------------------------------------------------------------------
# Public log
# ---------------------------------------------------------------------------

def log_append(msg: str) -> None:
    """Append a new public-log entry with the current UTC timestamp."""
    dt = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _lock:
        _conn.execute("INSERT INTO public_log (dt, msg) VALUES (?, ?)", (dt, msg))
        _conn.commit()


def log_read_since(seconds: int) -> list[dict]:
    """Return all public-log entries from the last *seconds* seconds, oldest first."""
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _lock:
        rows = _conn.execute(
            "SELECT dt, msg FROM public_log WHERE dt >= ? ORDER BY id",
            (cutoff,)
        ).fetchall()
    return [{"dt": r["dt"], "msg": r["msg"]} for r in rows]


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

def maintenance_create(label: str, added: list[str], purged: list[str]) -> int:
    """Insert a new maintenance record and return its id."""
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _lock:
        cur = _conn.execute(
            "INSERT INTO maintenance (label, started_at, added, purged, renewed)"
            " VALUES (?, ?, ?, ?, '[]')",
            (label, started_at, json.dumps(added), json.dumps(purged)),
        )
        _conn.commit()
        return cur.lastrowid


def maintenance_complete(maintenance_id: int) -> dict:
    """Mark a maintenance record as completed and return the full row."""
    completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _lock:
        _conn.execute(
            "UPDATE maintenance SET completed_at = ? WHERE id = ?",
            (completed_at, maintenance_id),
        )
        _conn.commit()
        row = _conn.execute(
            "SELECT * FROM maintenance WHERE id = ?", (maintenance_id,)
        ).fetchone()
    return dict(row)


def maintenance_add_renewed(maintenance_id: int, folder: str) -> None:
    """Append a folder name to the renewed JSON list of a maintenance record."""
    with _lock:
        row = _conn.execute(
            "SELECT renewed FROM maintenance WHERE id = ?", (maintenance_id,)
        ).fetchone()
        if row is None:
            return
        renewed = json.loads(row["renewed"])
        renewed.append(folder)
        _conn.execute(
            "UPDATE maintenance SET renewed = ? WHERE id = ?",
            (json.dumps(renewed), maintenance_id),
        )
        _conn.commit()


def maintenance_get_incomplete() -> list[dict]:
    """Return all maintenance records that have not yet been completed."""
    with _lock:
        rows = _conn.execute(
            "SELECT * FROM maintenance WHERE completed_at IS NULL ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Tilesets
# ---------------------------------------------------------------------------

def tileset_upsert(folder: str, run_str: str, step: int, target_str: str) -> bool:
    """Insert a new pending tileset entry. No-op if the folder already exists.
    Returns True if a new row was inserted."""
    queued_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _lock:
        cur = _conn.execute(
            "INSERT OR IGNORE INTO tilesets (folder, run_str, step, target_str, status, queued_at)"
            " VALUES (?, ?, ?, ?, 'pending', ?)",
            (folder, run_str, step, target_str, queued_at),
        )
        _conn.commit()
        return cur.rowcount > 0


def tileset_claim_pending() -> dict | None:
    """Atomically claim one pending tileset for processing.

    Selects the oldest pending row, flips its status to 'fetching', and returns
    it as a dict (including maintenance_id).  Returns None if the queue is empty.
    """
    with _lock:
        row = _conn.execute(
            "SELECT folder, run_str, step, target_str, maintenance_id FROM tilesets"
            " WHERE status = 'pending' ORDER BY rowid LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        _conn.execute(
            "UPDATE tilesets SET status = 'fetching' WHERE folder = ?",
            (row["folder"],),
        )
        _conn.commit()
        return dict(row)


def tileset_set_ready(folder: str, size: int) -> None:
    """Mark a tileset as ready and record its size."""
    completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _lock:
        _conn.execute(
            "UPDATE tilesets SET status = 'ready', size = ?, completed_at = ?"
            " WHERE folder = ?",
            (size, completed_at, folder),
        )
        _conn.commit()


def tileset_set_status(folder: str, status: str) -> None:
    """Update the status of a tileset entry."""
    with _lock:
        _conn.execute(
            "UPDATE tilesets SET status = ? WHERE folder = ?",
            (status, folder),
        )
        _conn.commit()


def tileset_set_maintenance(folder: str, maintenance_id: int) -> None:
    """Link a tileset entry to a maintenance record."""
    with _lock:
        _conn.execute(
            "UPDATE tilesets SET maintenance_id = ? WHERE folder = ?",
            (maintenance_id, folder),
        )
        _conn.commit()


def tileset_delete(folder: str) -> None:
    """Remove a tileset entry from the database."""
    with _lock:
        _conn.execute("DELETE FROM tilesets WHERE folder = ?", (folder,))
        _conn.commit()


def tileset_get_all(status: str | None = None) -> list[dict]:
    """Return all tileset entries, optionally filtered by status."""
    with _lock:
        if status is None:
            rows = _conn.execute("SELECT * FROM tilesets ORDER BY rowid").fetchall()
        else:
            rows = _conn.execute(
                "SELECT * FROM tilesets WHERE status = ? ORDER BY rowid",
                (status,),
            ).fetchall()
    return [dict(r) for r in rows]


def tileset_get_size() -> int:
    """Return the total size in bytes of all ready tilesets."""
    with _lock:
        row = _conn.execute(
            "SELECT COALESCE(SUM(size), 0) AS total FROM tilesets WHERE status = 'ready'"
        ).fetchone()
    return int(row["total"])


def tileset_count_pending() -> int:
    """Return the number of pending tileset entries."""
    with _lock:
        row = _conn.execute(
            "SELECT COUNT(*) AS cnt FROM tilesets WHERE status = 'pending'"
        ).fetchone()
    return int(row["cnt"])


def tileset_count_active_for_maintenance(maintenance_id: int) -> int:
    """Return the number of pending or fetching tilesets linked to a maintenance record."""
    with _lock:
        row = _conn.execute(
            "SELECT COUNT(*) AS cnt FROM tilesets"
            " WHERE maintenance_id = ? AND status IN ('pending', 'fetching')",
            (maintenance_id,),
        ).fetchone()
    return int(row["cnt"])
