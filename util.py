from math import floor
import sys

# MAX_ALTITUDE = 22500.0  # DWD ICON-D2 maximum altitude
MAX_ALTITUDE = 14000.0  # Sensible maximum altitude

def report_progress(stage, detail, percent):
    """
    Helper function to print structured progress.
    - If output is a terminal (TTY), it prints a nice, single-line status.
    - If output is a pipe (e.g., to the server), it prints machine-readable text.
    """
    percent = int(floor(percent+0.00001))
    if sys.stdout.isatty():
        progress_bar = f"[{percent:3d}%]"
        formatted_stage = f"{stage:<14}"
        sys.stdout.write(f"\r{progress_bar} {formatted_stage}: {detail}\x1b[K")

        if percent == 100:
            sys.stdout.write("\n")
    else:
        print(f"PROGRESS::{stage}::{detail}::{percent}")

    sys.stdout.flush()
