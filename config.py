# Daily UTC time (HH:MM) at which the auto-build and archive tasks run.
# The same tasks also run once immediately on server startup.
auto_build_time = "02:30"

# Number of days after which a time slot becomes eligible for archiving.
# Slots older than this threshold are pruned, except for the hours listed below.
archive_threshold_days = 7

# UTC hours (0-23) to retain per day once it exceeds archive_threshold_days.
# All other hours for those days are deleted.
# Example: [0, 12] keeps midnight and noon.
archive_keep_hours = [12]
