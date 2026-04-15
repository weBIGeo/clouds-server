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
from math import floor
import json
import os
import re
import sys
import threading
from datetime import datetime, timezone


# NOTE: The version is intentionally kept only in README.md. This function extracts it.
# I know its weird, but its one single line of truth - and i keep forgetting to bump it.
def read_version() -> str:
    try:
        readme = os.path.join(os.path.dirname(__file__), "README.md")
        with open(readme, encoding="utf-8") as f:
            m = re.search(r"img\.shields\.io/badge/version-([^-]+)-", f.read())
            if m:
                return m.group(1)
    except Exception:
        pass
    return "unknown"

# MAX_ALTITUDE = 22500.0  # DWD ICON-D2 maximum altitude
MAX_ALTITUDE = 14000.0  # Sensible maximum altitude

# Each section's (start_offset, weight) as fractions summing to 1.0.
_SECTION_WEIGHTS: dict[str, tuple[float, float]] = {
    "download":       (0.0,  0.30),
    "upsampling":     (0.30, 0.30),
    "lod_generation": (0.60, 0.30),
    "shadows":        (0.90, 0.10),
}


def report_progress(stage: str, detail: str, percent: float) -> None:
    """
    Helper function to print structured progress.
    - percent is the section-local progress (0-100).
    - When piped to the server, the emitted percent is the overall job progress
      based on the section weights defined in _SECTION_WEIGHTS.
    - If output is a terminal (TTY), it prints a nice, single-line status with
      the section-local percent.
    """
    section_percent = int(floor(percent + 0.00001))
    if sys.stdout.isatty():
        progress_bar = f"[{section_percent:3d}%]"
        formatted_stage = f"{stage:<14}"
        sys.stdout.write(f"\r{progress_bar} {formatted_stage}: {detail}\x1b[K")
        if section_percent == 100:
            sys.stdout.write("\n")
    else:
        offset, weight = _SECTION_WEIGHTS.get(stage, (0.0, 1.0))
        overall = int(offset * 100 + weight * section_percent)
        print(f"PROGRESS::{stage}::{detail}::{overall}")

    sys.stdout.flush()


class PublicLog:
    """Append-only public event log stored as JSONL (public-log.json).

    Each entry is a JSON object on its own line:
      {"dt": "2026-04-15T12:00:00Z", "msg": "..."}
    """

    def __init__(self, path: str = "public-log.json"):
        self._path = path
        self._lock = threading.Lock()

    def append(self, message: str) -> None:
        """Append a new entry with the current UTC time."""
        entry = {
            "dt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "msg": message,
        }
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def read_since(self, seconds: int) -> list[dict]:
        """Return all entries from the last *seconds* seconds, oldest first."""
        if not os.path.exists(self._path):
            return []
        cutoff = datetime.now(timezone.utc).timestamp() - seconds
        results = []
        with self._lock:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        dt = datetime.strptime(entry["dt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                        if dt.timestamp() >= cutoff:
                            results.append(entry)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        return results
