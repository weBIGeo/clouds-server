# Directory where generated tile sets are stored.
output_dir = "tiles_output"

# Port the Flask server listens on.
port = 8000

# If True, downloaded GRIB files are kept on disk after processing.
keep_gribs = False

# Number of parallel workers for downloading GRIB files from DWD.
download_workers = 32

# UTC times (HH:MM) at which the auto-build and archive tasks run.
# The same tasks also run once immediately on server startup.
# Default: 30 minutes after each ICON-D2 run (every 3 hours).
auto_build_time = ["00:30", "03:30", "06:30", "09:30", "12:30", "15:30", "18:30", "21:30"]

# Number of days after which a time slot becomes eligible for archiving.
# Slots older than this threshold are pruned, except for the hours listed below.
archive_threshold_days = 7

# UTC hours (0-23) to retain per day once it exceeds archive_threshold_days.
# All other hours for those days are deleted.
# Example: [0, 12] keeps midnight and noon.
archive_keep_hours = [12]
