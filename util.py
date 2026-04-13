from math import floor
import sys

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
