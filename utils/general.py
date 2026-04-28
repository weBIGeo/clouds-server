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
import os
import re
import sys


# NOTE: The version is intentionally kept only in README.md. This function extracts it.
# I know its weird, but its one single line of truth - and i keep forgetting to bump it.
def read_version() -> str:
    try:
        readme = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
        with open(readme, encoding="utf-8") as f:
            m = re.search(r"img\.shields\.io/badge/version-([^-]+)-", f.read())
            if m:
                return m.group(1)
    except Exception:
        pass
    return "unknown"


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

