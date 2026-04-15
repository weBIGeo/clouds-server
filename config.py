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

# Directory where generated tile sets are stored.
tile_cache_dir = "tiles_output"

# Maximum total size of the tile cache directory. Processing is skipped when exceeded.
tile_cache_max_size = 100 * 1000 * 1000 * 1000  # 100 GB

# Host and port the server listens on.
# Use "0.0.0.0" to accept connections from any network interface.
host = "127.0.0.1"
port = 8000

# Logging level. (e.g. DEBUG, INFO, WARNING, ERROR, CRITICAL)
log_level = "DEBUG"


# NOTE: Logs for the tile creation progress are stored in the respective tile folders. The following values
# only affect the server's own console log output.
# Log file path. Set to empty string to disable file logging.
log_file = "latest.log"
# Log file rotation: maximum size per file in bytes and number of backup files to keep.
log_file_max_bytes = 5 * 1024 * 1024  # 5 MB
log_file_backup_count = 3

# Per-logger level overrides. Use this to silence noisy third-party libraries
log_level_overrides = {
    "waitress": "ERROR",
    "filelock": "ERROR",
    "urllib3":  "ERROR",
    "numba":    "WARNING",
}

# If True, downloaded GRIB files are kept on disk after processing.
# NOTE: Use only for debug purposes. Those files are very large
keep_gribs = False

# Number of parallel workers for downloading GRIB files from DWD.
download_workers = 16

# Number of tile-set jobs that can be processed concurrently.
# NOTE: Each job depending on the step uses many threads/GPU internally, thats why in worker.py we use
# several file locks to actually serialize the work per step again. Therefore A value above 5 might not make
# a lot of sense and will probably not even affect the processing speed positively.
worker_threads = 3

# Unified scheme controlling which tile sets are fetched and which are purged.
# Rules are checked top-to-bottom
tile_retention_policy = [
    { # Remove all tiles that are older than a year
        "before": -364,
        "hours": [],
        "mode": "purge_only"
    },
    { # For tiles older than a week only keep the one at noon
        "before": -7,
        "hours": [12],
        "mode": "purge_only"
    },
    { # For tiles in the last week keep the ones with no step (the most accurate ones)
        "before": 0,
        "hours": [0, 3, 6, 9, 12, 15, 18, 21],
        "mode": "fetch_and_purge"
    },
    { # At the current day we want a different set of tiles for each hour
        "before": 1,
        "hours": list(range(24)),
        "mode": "fetch_only"
    },
    {   # For the next two days (forecasts) only fetch morning, noon, evening
        # NOTE: Increasing this would mean a lot of refetches each auto_build since
        # its very likely that a "better" model exists for all forecast tile_sets
        "before": 3,
        "hours": [6, 12, 18],
        "mode": "fetch_only"
    },
]

# UTC times (HH:MM) at which the auto-build and archive tasks run.
# The same tasks also run once immediately on server startup.
# NOTE: ICON-D2 runs are every 3 hours starting at 00:00, but it takes some time until they are available
auto_build_time = ["00:30", "07:30", "16:30"]

# When True, no new tiles are fetched or generated and no old tiles are purged.
# Only tiles that already exist on disk will be served.
only_serve = False
