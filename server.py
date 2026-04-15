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
import sys
import shutil
import subprocess
import threading
import re
import time
import logging
import config
import log_config
import util
import urllib.request
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from waitress import serve

logger = logging.getLogger("server")

VERSION = util.read_version()
public_log = util.PublicLog()

app = Flask(__name__)
CORS(app)

pending_tasks = {}  # target_time_id -> (run_dt, step)
pending_tasks_lock = threading.Lock()
pending_tasks_ready = threading.Event()
active_tasks = set()
processing_lock = threading.Lock()
task_progress = {}
next_maintenance: datetime | None = None
maintenance_batches: list[dict] = []
maintenance_batches_lock = threading.Lock()


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
    return os.path.join(os.path.abspath(config.output_dir), folder_name), folder_name


def scan_existing_folders():
    results = {}
    base_dir = os.path.abspath(config.output_dir)
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




def worker_output_reader(process, task_key, log_file_path):
    """Reads stdout from a worker process and updates the shared progress dict."""
    with open(log_file_path, "w", encoding="utf-8") as f:
        for line in iter(process.stdout.readline, ""):
            f.write(line)
            f.flush()
            line = line.strip()
            if line.startswith("PROGRESS::"):
                try:
                    _, stage, detail, percent_str = line.split("::", 3)
                    percent = int(percent_str)
                    progress = {"stage": stage, "detail": detail, "percent": percent}
                    with processing_lock:
                        if task_key in task_progress:
                            task_progress[task_key].update(progress)
                except (ValueError, IndexError):
                    pass

    process.stdout.close()


def worker_loop():
    while True:
        pending_tasks_ready.wait()

        with pending_tasks_lock:
            if not pending_tasks:
                pending_tasks_ready.clear()
                continue
            target_id, (run_dt, step) = next(iter(pending_tasks.items()))
            del pending_tasks[target_id]
            queue_depth = len(pending_tasks)
            if not pending_tasks:
                pending_tasks_ready.clear()

        run_str = run_dt.strftime("%Y%m%d%H")
        task_key = (run_str, step)

        folder_name = f"{run_str}_{step:03d}"
        output_dir = os.path.join(os.path.abspath(config.output_dir), folder_name)
        os.makedirs(output_dir, exist_ok=True)

        invalid_path = os.path.join(output_dir, "invalid")
        open(invalid_path, "w").close()

        log_path = os.path.join(output_dir, "latest.log")

        cmd = [
            sys.executable,
            "worker.py",
            "--run",
            run_str,
            "--step",
            str(step),
            "--out",
            output_dir,
        ]

        if config.keep_gribs:
            cmd.append("--keep-gribs")

        with processing_lock:
            task_progress[task_key] = {
                "status": "pending",
                "stage": "initializing",
                "detail": "starting worker process",
                "percent": 0,
            }

        logger.info(f"Processing: run {run_str} +{step}h (queue: {queue_depth} remaining)")
        start_time = time.monotonic()
        success = False
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            reader_thread = threading.Thread(
                target=worker_output_reader,
                args=(process, task_key, log_path),
                daemon=True,
            )
            reader_thread.start()

            process.wait()
            reader_thread.join()

            if process.returncode != 0:
                logger.error(f"Worker for {task_key} failed (exit {process.returncode}). See {log_path}")
            else:
                os.remove(invalid_path)
                elapsed = time.monotonic() - start_time
                logger.info(f"Done: run {run_str} +{step}h ({elapsed:.1f}s)")
                success = True

        except Exception as e:
            logger.error(f"Error launching worker: {e}")
        finally:
            with processing_lock:
                task_progress.pop(task_key, None)

        # Remove any stale folders for the same target time
        stale_removed = []
        if success:
            target_dt = run_dt + timedelta(hours=step)
            base_dir = os.path.abspath(config.output_dir)
            for name in os.listdir(base_dir):
                if name == folder_name:
                    continue
                path = os.path.join(base_dir, name)
                if not os.path.isdir(path) or "_" not in name:
                    continue
                try:
                    r2, s2 = name.split("_", 1)
                    if datetime.strptime(r2, "%Y%m%d%H") + timedelta(hours=int(s2)) == target_dt:
                        shutil.rmtree(path, ignore_errors=True)
                        logger.debug(f"Removed stale folder: {name}")
                        stale_removed.append(name)
                except ValueError:
                    continue

        if stale_removed:
            with maintenance_batches_lock:
                for batch in maintenance_batches:
                    if task_key in batch["task_keys"]:
                        batch["renewed"].extend(stale_removed)
                        break

        _check_maintenance_completion()



@app.route("/", methods=["GET"])
def index():
    return send_from_directory("docs", "index.html")


@app.route("/status", methods=["GET"])
def server_status():
    with pending_tasks_lock:
        queued = sorted(f"{run_dt.strftime('%Y%m%d%H')}_{step:03d}" for run_dt, step in pending_tasks.values())

    with processing_lock:
        active = {
            f"{run_str}_{step:03d}": dict(progress)
            for (run_str, step), progress in task_progress.items()
        }

    is_working = bool(queued or active)

    return jsonify({
        "version": VERSION,
        "status": "working" if is_working else "idle",
        "next_maintenance": next_maintenance.strftime("%Y-%m-%dT%H:%M:00Z") if next_maintenance else None,
        "active": active,
        "queued": queued,
    })


@app.route("/available", methods=["GET"])
def list_available():
    disk_state = scan_existing_folders()
    items = []
    for time_id, entry in disk_state.items():
        if entry["ready"]:
            items.append({
                "id": time_id,
                "status": "ready",
                "path": f"/{entry['folder']}/",
                "run": entry["run"].strftime("%Y%m%d%H"),
                "step": entry["step"],
            })
    items.sort(key=lambda x: x["id"])
    return jsonify({"items": items})


@app.route("/log", methods=["GET"])
def get_public_log():
    try:
        since = int(request.args.get("since", 7 * 24 * 3600))
    except (ValueError, TypeError):
        since = 7 * 24 * 3600
    return jsonify({"entries": public_log.read_since(since)})


@app.route("/<path:filename>")
def serve_tiles(filename):
    """
    Handles file serving with path rewriting:
    1. Tiles:  /{folder}/tiles/{z}/{x}/{y}.ktx2 -> /{folder}/tile_{z}_{x}_{y}.ktx2
    2. Shadow: /{folder}/shadow.ktx2             -> /{folder}/shadow.ktx2
    """

    # 1. Tile Route
    # Pattern: run_step/tiles/z/x/y.ktx2
    tile_match = re.match(r"^([^/]+)/tiles/(\d+)/(\d+)/(\d+)\.ktx2$", filename)
    if tile_match:
        folder, z, x, y = tile_match.groups()
        real_filename = f"tile_{z}_{x}_{y}.ktx2"

        # We send from the specific run_step folder
        return send_from_directory(
            os.path.join(os.path.abspath(config.output_dir), folder), real_filename
        )

    # 2. Shadow Route
    # Pattern: run_step/shadow.ktx2
    shadow_match = re.match(r"^([^/]+)/shadow\.ktx2$", filename)
    if shadow_match:
        folder = shadow_match.group(1)
        return send_from_directory(
            os.path.join(os.path.abspath(config.output_dir), folder), "shadow.ktx2"
        )

    return ("Forbidden", 403)


def auto_build_all():
    """Queue generation for configured time slots across past, current, and future days."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today = now.date()

    # Build targets from all fetch_and_purge rules in the scheme
    targets = []
    scheme = config.tile_retention_policy
    for i, rule in enumerate(scheme):
        if rule["mode"] not in ("fetch_and_purge", "fetch_only"):
            continue
        lower = scheme[i - 1]["before"] if i > 0 else -9999
        for day_offset in range(lower, rule["before"]):
            day = today + timedelta(days=day_offset)
            for h in rule["hours"]:
                targets.append(datetime(day.year, day.month, day.day, h))

    disk_state = scan_existing_folders()

    added = 0
    queued: list[tuple[str, tuple[str, int]]] = []
    for target_time in targets:
        time_id = target_time.strftime("%Y%m%d%H")
        best_run, best_step = get_best_run_and_step(target_time)
        if not best_run:
            continue

        task_key = (best_run.strftime("%Y%m%d%H"), best_step)

        with processing_lock:
            if task_key in task_progress:
                continue

        disk_entry = disk_state.get(time_id)
        if disk_entry and disk_entry["ready"] and disk_entry["run"] == best_run:
            continue

        with pending_tasks_lock:
            if pending_tasks.get(time_id) is not None:
                continue  # already queued, do not replace
            pending_tasks[time_id] = (best_run, best_step)

        folder = f"{best_run.strftime('%Y%m%d%H')}_{best_step:03d}"
        queued.append((folder, task_key))
        added += 1
        logger.debug(f"AutoBuild: {folder} (target {time_id})")

    logger.info(f"AutoBuild done: {added} added")
    # Signal workers only after all tasks are queued so they see correct queue depths.
    with pending_tasks_lock:
        if pending_tasks:
            pending_tasks_ready.set()
    return queued


def purge_old_data():
    """Remove invalid dirs and apply the tile_retention_policy rules."""
    base_dir = os.path.abspath(config.output_dir)
    if not os.path.exists(base_dir):
        return []

    purged: list[str] = []
    for entry in scan_existing_folders().values():
        path = os.path.join(base_dir, entry["folder"])

        with processing_lock:
            if (entry["run"].strftime("%Y%m%d%H"), entry["step"]) in task_progress:
                continue

        if not entry["ready"]:
            logger.debug(f"Purge: {entry['folder']} (invalid/incomplete)")
            shutil.rmtree(path, ignore_errors=True)
            purged.append(entry["folder"])
            continue

        target_dt = entry["run"] + timedelta(hours=entry["step"])
        rule = get_scheme_rule(target_dt)

        if rule is None:
            continue  # beyond all rules, leave untouched

        if rule["mode"] != "fetch_only" and target_dt.hour not in rule["hours"]:
            logger.debug(f"Purge: {entry['folder']} (hour {target_dt.hour} not in retention policy hours {rule['hours']})")
            shutil.rmtree(path, ignore_errors=True)
            purged.append(entry["folder"])

    logger.info(f"Purge: removed {len(purged)} tile set(s)")
    return purged


def _check_maintenance_completion():
    """Called by worker_loop after each task finishes. Logs and removes any batch whose tasks are all done."""
    with pending_tasks_lock:
        pending_keys = {(rd.strftime("%Y%m%d%H"), s) for rd, s in pending_tasks.values()}
    with processing_lock:
        active_keys = set(task_progress.keys())
    still_running = pending_keys | active_keys

    with maintenance_batches_lock:
        done = [b for b in maintenance_batches if not (b["task_keys"] & still_running)]
        for b in done:
            maintenance_batches.remove(b)

    for batch in done:
        elapsed     = (time.monotonic() - batch["start_time"]) / 60
        added_str   = ", ".join(batch["folder_names"]) or "none"
        purged_str  = ", ".join(batch["purged"])        or "none"
        renewed_str = ", ".join(batch["renewed"])       or "none"
        logger.info("Maintenance completed in {elapsed:.1f}min")
        public_log.append(f"Maintenance completed in {elapsed:.1f}min:\n - {len(batch['folder_names'])} added ({added_str})\n - {len(batch['purged'])} purged ({purged_str})\n - {len(batch['renewed'])} renewed ({renewed_str})")


def _run_maintenance():
    start_time = time.monotonic()
    logger.info("Scheduler: running maintenance")
    purged_folders = purge_old_data()
    queued = auto_build_all()
    if queued:
        with maintenance_batches_lock:
            maintenance_batches.append({
                "task_keys":    {tk for _, tk in queued},
                "folder_names": [f  for f,  _ in queued],
                "purged":       purged_folders,
                "renewed":      [],
                "start_time":   start_time,
            })
    else:
        elapsed    = (time.monotonic() - start_time) / 60
        purged_str = ", ".join(purged_folders) or "none"
        summary = f"Maintenance completed in {elapsed:.1f}min:"
        logger.info(summary)
        public_log.append(f"{summary}\n"
                          f" - 0 added (none)\n"
                          f" - {len(purged_folders)} purged ({purged_str})\n"
                          f" - 0 renewed (none)")


def scheduler_loop():
    """Runs _run_maintenance on startup and at each configured auto_build_time."""
    global next_maintenance
    _run_maintenance()

    while True:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        times = [now.replace(hour=int(e[:2]), minute=int(e[3:]), second=0, microsecond=0)
                 for e in config.auto_build_time]
        next_maintenance = next((t for t in times if t > now), times[0] + timedelta(days=1))
        logger.info(f"Scheduler: next maintenance at {next_maintenance.strftime('%Y-%m-%d %H:%M')} UTC")
        time.sleep((next_maintenance - now).total_seconds())
        _run_maintenance()


if __name__ == "__main__":
    log_config.setup_logging(log_file=config.log_file)
    log_config.print_logo()
    public_log.append(f"Server started v{VERSION}")
    msg = f" === weBIGeo Cloud Server v{VERSION} started === "
    sep = " " + "=" * (len(msg) - 2) + " "
    logger.info(sep)
    logger.info(msg)
    logger.info(sep)
    output_dir = os.path.abspath(config.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not config.only_serve:
        for i in range(config.worker_threads):
            threading.Thread(target=worker_loop, daemon=True, name=f"WorkerThread-{i}").start()
        threading.Thread(target=scheduler_loop, daemon=True, name="SchedulerThread").start()
    else:
        logger.info("only_serve = True -> no background scheduler and worker started")

    logger.info(f"Starting waitress server on http://{config.host}:{config.port}")
    serve(app, host=config.host, port=config.port)
