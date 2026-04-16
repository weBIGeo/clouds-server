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

import os
import re
import shutil
import threading
import urllib.request
import logging
import config
import db
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("tile_cache")

# Cache of available steps per run_str (YYYYMMDDHH) discovered from the DWD
# listing at grib/{HH}/clc/. Each entry maps run_str -> {"steps": set(int), "ts": datetime, "status": "success" or "fail"}
DWD_RUN_CACHE = {}
DWD_RUN_CACHE_LOCK = threading.Lock()
# When a listing fetch fails, cache that failure for this many seconds
DWD_FAIL_TTL = 60
# ICON-D2 only produces runs at 00, 03, 06, 09..
DWD_RUN_INTERVAL = 3


def get_scheme_rule(target_dt):
    """Return the first matching scheme rule for target_dt, or None."""
    today = datetime.now(timezone.utc).replace(tzinfo=None).date()
    day_offset = (target_dt.date() - today).days
    for rule in config.tile_retention_policy:
        if day_offset < rule["before"]:
            return rule
    return None


def _scheme_fetch_window():
    """Return (min_day_offset, max_day_offset) covering all fetch modes (fetch_and_purge + fetch_only)."""
    scheme = config.tile_retention_policy
    fetch_indices = [i for i, r in enumerate(scheme) if r["mode"] in ("fetch_and_purge", "fetch_only")]
    if not fetch_indices:
        return 0, 0
    first_i, last_i = fetch_indices[0], fetch_indices[-1]
    lower = scheme[first_i - 1]["before"] if first_i > 0 else -9999
    upper = scheme[last_i]["before"] - 1
    return lower, upper


def fetch_run_steps(run_str, timeout=10):
    """Return a set of available step integers for the given run_str (YYYYMMDDHH).

    If fetching or parsing fails, returns an empty set.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today_dt = datetime(now.year, now.month, now.day)
    lower_offset, upper_offset = _scheme_fetch_window()

    # Window bounds for relevance
    min_time = today_dt + timedelta(days=lower_offset)
    max_time = today_dt + timedelta(days=upper_offset + 1)

    # Parse run datetime; if invalid, return empty
    try:
        run_dt = datetime.strptime(run_str, "%Y%m%d%H")
    except Exception:
        return set()

    # If the run cannot produce any target inside our window, do not fetch
    if run_dt > max_time:
        return set()
    if run_dt + timedelta(days=upper_offset + 1) < min_time:
        return set()

    # Check cache
    with DWD_RUN_CACHE_LOCK:
        cached = DWD_RUN_CACHE.get(run_str)
        if cached:
            status = cached.get("status")
            if status == "success":
                return set(cached.get("steps", set()))
            if status == "fail":
                age = (now - cached.get("ts", now)).total_seconds()
                if age < DWD_FAIL_TTL:
                    return set()

    run_hh = run_str[-2:]
    listing_url = f"https://opendata.dwd.de/weather/nwp/icon-d2/grib/{run_hh}/clc/"

    try:
        with urllib.request.urlopen(listing_url, timeout=timeout) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="ignore")
    except Exception:
        # Cache failure to debounce repeated failed fetches
        with DWD_RUN_CACHE_LOCK:
            DWD_RUN_CACHE[run_str] = {"steps": set(), "status": "fail", "ts": datetime.now(timezone.utc).replace(tzinfo=None)}
        return set()

    # Regex to find filenames for this run_str and capture the step (three digits)
    pattern = fr'href="icon-d2_germany_regular-lat-lon_model-level_{run_str}_(\d{{3}})_1_clc.grib2.bz2"'
    found = re.findall(pattern, text)

    steps = set()
    for s in found:
        try:
            steps.add(int(s))
        except ValueError:
            pass

    with DWD_RUN_CACHE_LOCK:
        if steps:
            DWD_RUN_CACHE[run_str] = {"steps": steps, "status": "success", "ts": datetime.now(timezone.utc).replace(tzinfo=None)}
        else:
            DWD_RUN_CACHE[run_str] = {"steps": set(), "status": "fail", "ts": datetime.now(timezone.utc).replace(tzinfo=None)}

    return steps


def is_dwd_available(run_dt, step, timeout=10):
    """Return True if the given run_dt+step exists on DWD (uses `fetch_run_steps`)."""
    run_str = run_dt.strftime("%Y%m%d%H")
    steps = fetch_run_steps(run_str, timeout=timeout)
    return step in steps


def get_best_run_and_step(target_time):
    """Return the best (most recent) available (run_datetime, step) for target_time.

    Steps backwards through DWD_RUN_INTERVAL-aligned run times only, up to the
    maximum fetch window, and returns the first one published on DWD.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    _, upper_offset = _scheme_fetch_window()
    # Start at the first aligned run at or before target_time, then stride by interval.
    first_step = target_time.hour % DWD_RUN_INTERVAL
    for step in range(first_step, upper_offset * 24 + 1, DWD_RUN_INTERVAL):
        run_time = target_time - timedelta(hours=step)

        if run_time > now:
            continue

        if is_dwd_available(run_time, step):
            return run_time, step

    return None, None


def get_folder_path(run_time, step):
    folder_name = f"{run_time.strftime('%Y%m%d%H')}_{step:03d}"
    return os.path.join(os.path.abspath(config.tile_cache_dir), folder_name), folder_name


def scan_existing_folders():
    results = {}
    base_dir = os.path.abspath(config.tile_cache_dir)
    if not os.path.exists(base_dir):
        return results

    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and "_" in name:
            try:
                r_str, s_str = name.split("_")
                run_dt = datetime.strptime(r_str, "%Y%m%d%H")
                step = int(s_str)
                target_dt = run_dt + timedelta(hours=step)
                target_str = target_dt.strftime("%Y%m%d%H")

                is_ready = not os.path.isfile(os.path.join(path, "invalid"))

                if target_str not in results or (run_dt > results[target_str]["run"]):
                    results[target_str] = {
                        "folder": name,
                        "run": run_dt,
                        "step": step,
                        "ready": is_ready,
                    }
            except ValueError:
                continue
    return results


def compute_folder_size(folder_path: str) -> int:
    """Return the total size in bytes of all files inside a single folder."""
    total = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def sync_from_disk() -> None:
    """Reconcile the tile_cache DB table with the actual filesystem on startup.

    Handles all restart-recovery cases so the DB is the authoritative source of
    truth before any worker threads begin.
    """
    base_dir = os.path.abspath(config.tile_cache_dir)
    os.makedirs(base_dir, exist_ok=True)

    disk_folders = scan_existing_folders()  # target_str -> entry dict
    # Build a flat map of folder_name -> disk entry for quick lookup
    disk_by_folder = {e["folder"]: e for e in disk_folders.values()}

    db_rows = {r["folder"]: r for r in db.tile_get_all()}

    # --- Reconcile DB rows against disk ---
    for folder, row in db_rows.items():
        path = os.path.join(base_dir, folder)
        disk = disk_by_folder.get(folder)

        if row["status"] == "fetching":
            if disk is None or not disk["ready"]:
                # Incomplete or missing — delete folder if present, requeue
                if os.path.exists(path):
                    shutil.rmtree(path, ignore_errors=True)
                    logger.info(f"sync_from_disk: removed orphaned folder {folder}, requeueing")
                else:
                    logger.info(f"sync_from_disk: fetching task {folder} has no folder, requeueing")
                db.tile_set_status(folder, "pending")
            else:
                # Folder is valid — server crashed after completion but before DB update
                size = compute_folder_size(path)
                db.tile_set_ready(folder, size, 0.0)
                logger.info(f"sync_from_disk: adopted completed folder {folder} ({size} bytes)")

        elif row["status"] == "pending":
            if disk is not None and disk["ready"]:
                # Folder is already valid on disk — promote to ready
                size = compute_folder_size(path)
                db.tile_set_ready(folder, size, 0.0)
                logger.info(f"sync_from_disk: promoted pending {folder} to ready ({size} bytes)")
            # pending with no folder on disk: leave as-is, workers will claim it

        elif row["status"] == "ready":
            if disk is None:
                # Folder was deleted externally — remove from DB
                db.tile_delete(folder)
                logger.info(f"sync_from_disk: removed DB entry for missing folder {folder}")

    # --- Adopt disk folders not known to DB ---
    for folder, entry in disk_by_folder.items():
        if folder in db_rows:
            continue
        path = os.path.join(base_dir, folder)
        if entry["ready"]:
            run_str = entry["run"].strftime("%Y%m%d%H")
            target_str = (entry["run"] + timedelta(hours=entry["step"])).strftime("%Y%m%d%H")
            size = compute_folder_size(path)
            db.tile_upsert(folder, run_str, entry["step"], target_str)
            db.tile_set_ready(folder, size, 0.0)
            logger.info(f"sync_from_disk: adopted unknown folder {folder} ({size} bytes)")
        else:
            # Invalid/incomplete folder with no DB record — discard
            shutil.rmtree(path, ignore_errors=True)
            logger.info(f"sync_from_disk: deleted orphaned invalid folder {folder}")
