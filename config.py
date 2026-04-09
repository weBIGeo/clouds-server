# Directory where generated tile sets are stored.
output_dir = "tiles_output"

# Port the Flask server listens on.
port = 8000

# If True, downloaded GRIB files are kept on disk after processing.
# NOTE: Use only for debug purposes. Those files are very large
keep_gribs = False

# Number of parallel workers for downloading GRIB files from DWD.
download_workers = 16

# Number of tile-set jobs that can be processed concurrently.
# Each job already uses many threads/GPU internally, so >1 only helps if you
# have spare resources (e.g. one job downloading while another is processing).
# NOTE: When >1, downloads are serialized so jobs naturally spread across
# different pipeline stages (download vs. processing vs. saving).
worker_threads = 3

# Unified scheme controlling which tile sets are fetched and which are purged.
# Rules are checked top-to-bottom; the first rule where the target's day offset
# (relative to today) is less than "before" wins.
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
