#############################################################################
# weBIGeo Clouds
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
        _conn.executescript("""
            CREATE TABLE IF NOT EXISTS public_log (
                id  INTEGER PRIMARY KEY AUTOINCREMENT,
                dt  TEXT NOT NULL,
                msg TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_public_log_dt ON public_log(dt);
        """)
        _conn.commit()


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
