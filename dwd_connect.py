import os
import requests
import bz2
import shutil
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from util import report_progress

# --- Configuration ---

# Base URL for DWD OpenData
BASE_URL = "https://opendata.dwd.de/weather/nwp/icon-d2/grib"

# Local cache directory
CACHE_DIR = "dwd_cache"

# Model Vertical Levels
# Levels 1 to 65 are prognostic. Level 66 is the surface interface for HHL.
MODEL_LEVELS = range(1, 66)
HHL_LEVELS = range(1, 67)


@dataclass
class VarConfig:
    dwd_name: str  # The shortname used in the URL/Filename (e.g., 'qc', 'hhl')
    level_type: str  # 'model-level', 'single-level', 'time-invariant'
    has_levels: bool  # True if it iterates over vertical levels
    levels: range  # The specific range of levels to fetch
    prefix: str = ""  # Some 2D vars have '2d_' prefix


# Declarative Variable Definitions
VAR_SPECS = {
    # --- Geometry (Invariant) ---
    "hhl": VarConfig("hhl", "time-invariant", True, HHL_LEVELS),
    "hsurf": VarConfig("hsurf", "time-invariant", True, [0]),
    # --- Thermodynamics (Model Levels) ---
    "p": VarConfig("p", "model-level", True, MODEL_LEVELS),
    "t": VarConfig("t", "model-level", True, MODEL_LEVELS),
    "w": VarConfig("w", "model-level", True, HHL_LEVELS), # Vertical velocity is on the half-levels (HHL)
    "u": VarConfig("u", "model-level", True, MODEL_LEVELS),
    "v": VarConfig("v", "model-level", True, MODEL_LEVELS),
    # --- Cloud Physics (Model Levels) ---
    "clc": VarConfig("clc", "model-level", True, MODEL_LEVELS),  # Cloud Cover
    "clct": VarConfig(
        "clct", "single-level", False, [0], prefix="2d_"
    ),  # Total Cloud Cover
    "clcl": VarConfig(
        "clcl", "single-level", False, [0], prefix="2d_"
    ),  # Low Cloud Cover
    "clcm": VarConfig(
        "clcm", "single-level", False, [0], prefix="2d_"
    ),  # Medium Cloud Cover
    "clch": VarConfig(
        "clch", "single-level", False, [0], prefix="2d_"
    ),  # High Cloud Cover
    "qc": VarConfig("qc", "model-level", True, MODEL_LEVELS),  # Cloud Water
    "qi": VarConfig("qi", "model-level", True, MODEL_LEVELS),  # Cloud Ice
    "qs": VarConfig("qs", "model-level", True, MODEL_LEVELS),  # Snow
    "qv": VarConfig("qv", "model-level", True, MODEL_LEVELS),  # Specific Humidity (Water Vapor)
    "qr": VarConfig("qr", "model-level", True, MODEL_LEVELS),  # Rain
    "tke": VarConfig("tke", "model-level", True, HHL_LEVELS),  # Turbulent Kinetic Energy
}


class DWDDownloader:
    def __init__(self, base_dir: str = CACHE_DIR):
        self.base_dir = base_dir
        self.session = requests.Session()

    def get_latest_run_time(self) -> datetime:
        """
        Determines the latest likely available run.
        ICON-D2 runs every 3 hours (00, 03, ... 21).
        Data is typically available ~2 hours after run time.
        """
        now = datetime.now(timezone.utc)
        # Go back 2 hours to be safe, then floor to nearest 3-hour block
        safe_time = now - timedelta(hours=2, minutes=15)
        hour = (safe_time.hour // 3) * 3
        return safe_time.replace(hour=hour, minute=0, second=0, microsecond=0)

    def _build_url_and_path(
        self, run_dt: datetime, step_hours: int, var_key: str, level: int = 0
    ) -> Tuple[str, str]:
        """
        Constructs the remote URL and the local target path.
        """
        config = VAR_SPECS[var_key]

        # Formatting
        run_hour_str = run_dt.strftime("%H")  # e.g., "09"
        date_str = run_dt.strftime("%Y%m%d%H")  # e.g., "2023102709"
        step_str = f"{step_hours:03d}"  # e.g., "005"

        # Invariant files behave differently (no step, no run hour in folder sometimes)
        if config.level_type == "time-invariant":
            # Pattern: .../time-invariant_2023102709_000_65_hhl.grib2.bz2
            # Note: Invariant files often use step 000 and the run time.
            lvl_str = f"_{level}" if config.has_levels else ""
            filename = (
                f"icon-d2_germany_regular-lat-lon_{config.level_type}_"
                f"{date_str}_000{lvl_str}_{config.dwd_name.lower()}.grib2.bz2"
            )
            remote_subdir = config.dwd_name.lower()
        else:
            # Pattern: .../model-level_2023102709_005_1_qc.grib2.bz2
            lvl_str = f"_{level}" if config.has_levels else ""
            prefix = config.prefix
            filename = (
                f"icon-d2_germany_regular-lat-lon_{config.level_type}_"
                f"{date_str}_{step_str}{lvl_str}_{prefix}{config.dwd_name.lower()}.grib2.bz2"
            )
            remote_subdir = config.dwd_name.lower()

        url = f"{BASE_URL}/{run_hour_str}/{remote_subdir}/{filename}"

        # Local Path: cache/2023102709/005/qc_65.grib2
        # We simplify local naming to avoid the massive DWD filenames
        lvl_suffix = f"_{level}" if config.has_levels else ""
        step_dir = "invariant" if config.level_type == "time-invariant" else step_str

        local_dir = os.path.join(self.base_dir, date_str, step_dir)
        local_filename = f"{var_key}{lvl_suffix}.grib2"
        local_path = os.path.join(local_dir, local_filename)

        return url, local_path

    def _download_single(self, url: str, local_path: str) -> bool:
        """
        Downloads, decompresses, and saves a single file.
        Returns True if successful (or already exists).
        """
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return True

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        temp_bz2 = local_path + ".bz2"

        try:
            with self.session.get(url, stream=True, timeout=30) as r:
                if r.status_code != 200:
                    # Some levels might not exist for some vars, or network error
                    print(f"    [404/Error] {url}")
                    return False

                with open(temp_bz2, "wb") as f:
                    shutil.copyfileobj(r.raw, f)

            # Decompress
            with bz2.BZ2File(temp_bz2, "rb") as f_in, open(local_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            # Cleanup
            os.remove(temp_bz2)
            return True

        except Exception as e:
            print(f"    [Exception] {url} -> {e}")
            if os.path.exists(temp_bz2):
                os.remove(temp_bz2)
            return False

    def fetch_variables(
        self, run_dt: datetime, step_hours: int, vars_to_fetch: List[str] = None
    ) -> Dict[str, List[str]]:
        """
        Main entry point. Downloads all requested variables for a specific run/step.
        Parallelized.
        """
        if vars_to_fetch is None:
            vars_to_fetch = list(VAR_SPECS.keys())

        tasks = []
        results = {v: [] for v in vars_to_fetch}

        print(
            f"--- Starting Download: Run {run_dt.strftime('%Y-%m-%d %H:00')} +{step_hours}h ---"
        )

        with ThreadPoolExecutor(max_workers=8) as executor:
            for var_key in vars_to_fetch:
                if var_key not in VAR_SPECS:
                    print(f"Warning: Unknown variable {var_key}")
                    continue

                config = VAR_SPECS[var_key]
                levels = config.levels if config.has_levels else [0]

                for level in levels:
                    url, path = self._build_url_and_path(
                        run_dt, step_hours, var_key, level
                    )

                    # Store expected path to return later
                    results[var_key].append(path)

                    # Submit task
                    future = executor.submit(self._download_single, url, path)
                    tasks.append(future)

            # Wait for completion and show progress
            total = len(tasks)
            completed = 0
            for _ in as_completed(tasks):
                completed += 1
                report_progress(
                    "download",
                    f"{completed}/{total} files processed",
                    completed / total * 100,
                )

        print("\n--- Download Complete ---")
        return results


# --- CLI for Testing ---
if __name__ == "__main__":
    downloader = DWDDownloader()

    # Determine Time
    latest_run = downloader.get_latest_run_time()
    forecast_step = 0  # +0 hours

    # Select Variables (Test with a subset to save time/bandwidth)
    # In a real run, you'd fetch all (None).
    test_vars = ["clc", "hsurf", "hhl"]

    # Fetch
    files = downloader.fetch_variables(latest_run, forecast_step, test_vars)

    # Verify
    print("Downloaded file manifest:")
    for k, paths in files.items():
        print(f"  {k}: {len(paths)} files (First: {paths[0] if paths else 'None'})")
