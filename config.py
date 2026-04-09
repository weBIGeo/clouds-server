# Directory where generated tile sets are stored.
output_dir = "tiles_output"

# Port the Flask server listens on.
port = 8000

# If True, downloaded GRIB files are kept on disk after processing.
keep_gribs = False

# Number of parallel workers for downloading GRIB files from DWD.
download_workers = 32

# How many past days to maintain tile sets for.
history_days = 1

# How many future days to generate forecasts for.
forecast_days = 2

# Hours of the day (UTC) to build for past days.
past_day_hours = [0, 3, 6, 9, 12, 15, 18, 21]

# Hours of the day (UTC) to build for the current day.
current_day_hours = list(range(0, 24))

# Hours of the day (UTC) to build for future days.
future_day_hours = [6, 12, 18]

# UTC times (HH:MM) at which the auto-build and archive tasks run.
# The same tasks also run once immediately on server startup.
# Default: 30 minutes after each ICON-D2 run (every 3 hours).
auto_build_time = ["00:30", "03:30", "06:30", "09:30", "12:30", "15:30", "18:30", "21:30"]

# Number of days after which a time slot becomes eligible for long-term purging.
# Slots older than this threshold are deleted, except for the hours listed below.
purge_threshold_days = 7

# UTC hours (0-23) to retain per day once it exceeds purge_threshold_days.
# All other hours for those days are deleted.
# Example: [0, 12] keeps midnight and noon.
purge_keep_hours = [12]
