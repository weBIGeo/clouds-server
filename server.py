import os
import sys
import shutil
import subprocess
import threading
import queue
import re
import time
import config
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

DWD_RUN_INTERVAL = 3
MAX_FORECAST_STEP = 48
HISTORY_WINDOW = 24

app = Flask(__name__)
CORS(app)

generation_queue = queue.Queue()
active_tasks = set()
processing_lock = threading.Lock()
task_progress = {}


# Cache of available steps per run_str (YYYYMMDDHH) discovered from the DWD
# listing at grib/{HH}/clc/. Each entry maps run_str -> {"steps": set(int), "ts": datetime, "status": "success" or "fail"}
DWD_RUN_CACHE = {}
DWD_RUN_CACHE_LOCK = threading.Lock()
# When a listing fetch fails, cache that failure for this many seconds
DWD_FAIL_TTL = 60

# Cache for get_best_run_and_step: maps time_id -> {"run": datetime or None, "step": int or None, "ts": datetime}
BEST_RUN_CACHE = {}
BEST_RUN_TTL = 60
BEST_RUN_CACHE_LOCK = threading.Lock()


def fetch_run_steps(run_str, timeout=10):
    """Return a set of available step integers for the given run_str (YYYYMMDDHH).

    If fetching or parsing fails, returns an empty set.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # Window bounds for relevance
    min_time = now - timedelta(hours=HISTORY_WINDOW)
    max_time = now + timedelta(hours=MAX_FORECAST_STEP)

    # Parse run datetime; if invalid, return empty
    try:
        run_dt = datetime.strptime(run_str, "%Y%m%d%H")
    except Exception:
        return set()

    # If the run cannot produce any target inside our window, do not fetch
    if run_dt > max_time:
        return set()
    if run_dt + timedelta(hours=MAX_FORECAST_STEP) < min_time:
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
    """Return the best available (run_datetime, step) for the given target_time.

    This function checks cached results for `target_time` (1-minute TTL) and
    otherwise inspects candidate runs stepping backwards from the target time
    (0..MAX_FORECAST_STEP hours). It only considers runs aligned to
    `DWD_RUN_INTERVAL` and not in the future.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    time_id = target_time.strftime("%Y%m%d%H")

    # Fast-path: cached result
    with BEST_RUN_CACHE_LOCK:
        entry = BEST_RUN_CACHE.get(time_id)
        if entry and (now - entry["ts"]).total_seconds() < BEST_RUN_TTL:
            return entry["run"], entry["step"]

    best = (None, None)

    for step in range(0, MAX_FORECAST_STEP + 1):
        run_time = target_time - timedelta(hours=step)

        # Must align to configured run interval and not be in the future
        if (run_time.hour % DWD_RUN_INTERVAL) != 0 or run_time > now:
            continue

        if is_dwd_available(run_time, step):
            best = (run_time, step)
            break

    with BEST_RUN_CACHE_LOCK:
        BEST_RUN_CACHE[time_id] = {"run": best[0], "step": best[1], "ts": datetime.now(timezone.utc).replace(tzinfo=None)}

    return best


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

                # Check for shadow map existence to confirm readiness
                is_ready = os.path.isfile(os.path.join(path, "shadow.ktx2"))

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


def cleanup_old_data():
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=HISTORY_WINDOW)

    folders = []
    base_dir = os.path.abspath(config.output_dir)
    if os.path.exists(base_dir):
        folders = [
            f
            for f in os.listdir(base_dir)
            if "_" in f and os.path.isdir(os.path.join(base_dir, f))
        ]

    for name in folders:
        try:
            r_str, s_str = name.split("_")
            run_dt = datetime.strptime(r_str, "%Y%m%d%H")
            step = int(s_str)
            target = run_dt + timedelta(hours=step)

            if target < cutoff:
                task_key = (r_str, step)
                if task_key in active_tasks:
                    continue
                shutil.rmtree(os.path.join(base_dir, name), ignore_errors=True)
        except:
            continue


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
                    print(f"[Worker {task_key[0]}+{task_key[1]}h] {stage}: {detail} ({percent}%)")
                except (ValueError, IndexError):
                    pass

    process.stdout.close()


def worker_loop():
    while True:
        task = generation_queue.get()
        if task is None:
            break

        run_dt, step = task
        run_str = run_dt.strftime("%Y%m%d%H")
        task_key = (run_str, step)

        cleanup_old_data()

        folder_name = f"{run_str}_{step:03d}"
        output_dir = os.path.join(os.path.abspath(config.output_dir), folder_name)
        os.makedirs(output_dir, exist_ok=True)

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

        queue_depth = generation_queue.qsize()
        print(f"[Server] Processing: run {run_str} +{step}h (queue: {queue_depth} remaining)")
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
                print(f"[Server] Worker for {task_key} failed (exit {process.returncode}). See {log_path}")
            else:
                print(f"[Server] Done: run {run_str} +{step}h")
                success = True

        except Exception as e:
            print(f"[Server] Error launching worker: {e}")
        finally:
            with processing_lock:
                task_progress.pop(task_key, None)

        # Remove any stale folders for the same target time
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
                        print(f"[Server] Removed stale folder: {name}")
                except ValueError:
                    continue

        generation_queue.task_done()


def get_slot_status(target_time):
    """
    Consolidated logic to get the status of a single time slot.
    This will be used by both /status and /available.
    """
    time_id = target_time.strftime("%Y%m%d%H")

    # 1. Determine the 'best' possible forecast for this time
    best_run, best_step = get_best_run_and_step(target_time)
    if not best_run:
        return {
            "id": time_id,
            "status": "error",
            "message": "Time is too far in the future",
        }

    best_run_str = best_run.strftime("%Y%m%d%H")

    slot_data = {
        "id": time_id,
        "status": "unknown",  # Default status if not found on disk or in queue
        "run": best_run_str,
        "step": best_step,
    }

    # 2. Check if it's currently being generated
    with processing_lock:
        task_key = (best_run_str, best_step)
        if task_key in task_progress:
            slot_data["status"] = "pending"
            slot_data["progress"] = task_progress[task_key]
            return slot_data

    # 3. Check if data exists on disk
    disk_state = scan_existing_folders()
    if time_id in disk_state:
        entry = disk_state[time_id]
        if entry["ready"]:
            slot_data["path"] = f"/{entry['folder']}/"
            slot_data["run"] = entry["run"].strftime("%Y%m%d%H")
            slot_data["step"] = entry["step"]
            # Is the data on disk the "best" possible data?
            if entry["run"] == best_run:
                slot_data["status"] = "ready"
            else:
                slot_data["status"] = "stale"

    return slot_data


@app.route("/status", methods=["GET"])
def get_status():
    """NEW: Endpoint to get the status of a single time slot."""
    time_str = request.args.get("time")
    if not time_str:
        return jsonify({"error": "Missing time parameter"}), 400

    try:
        target_time = datetime.strptime(time_str, "%Y%m%d%H")
    except ValueError:
        return jsonify({"error": "Invalid time format"}), 400

    # Check time window boundaries
    min_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=HISTORY_WINDOW)
    max_time = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=MAX_FORECAST_STEP)
    is_in_window = min_time <= target_time <= max_time

    if is_in_window:
        status = get_slot_status(target_time)
        return jsonify(status)

    # Outside the live window — check disk for historical data.
    time_id = target_time.strftime("%Y%m%d%H")
    disk_state = scan_existing_folders()
    entry = disk_state.get(time_id)

    if entry and entry["ready"]:
        slot_data = {
            "id": time_id,
            "status": "ready",
            "path": f"/{entry['folder']}/",
            "run": entry["run"].strftime("%Y%m%d%H"),
            "step": entry["step"],
        }
        return jsonify(slot_data)
    else:
        return (
            jsonify({"status": "error", "message": "Data not available for this time"}),
            404,
        )


@app.route("/generate", methods=["POST"])
def generate_request():
    time_str = request.args.get("time")
    if not time_str:
        return jsonify({"error": "Missing time"}), 400

    try:
        target_time = datetime.strptime(time_str, "%Y%m%d%H")
    except ValueError:
        return jsonify({"error": "Invalid format"}), 400

    best_run, best_step = get_best_run_and_step(target_time)
    if not best_run:
        return (
            jsonify({"status": "error", "message": "Forecast too far in future"}),
            400,
        )

    best_run_str = best_run.strftime("%Y%m%d%H")

    with processing_lock:
        task_key = (best_run_str, best_step)
        if task_key not in task_progress:
            # Add to queue only if not already being processed
            print(f"[Server] Queuing generation for {task_key}")
            generation_queue.put((best_run, best_step))
        else:
            print(f"[Server] Generation for {task_key} is already in progress.")

    # Respond with 202 Accepted, indicating the request was received.
    # The client should then use the /status endpoint to poll.
    return jsonify({"status": "pending", "message": "Generation has been queued"}), 202


@app.route("/available", methods=["GET"])
def list_available():
    now = datetime.now(timezone.utc).replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
    valid_slots = []
    processed_ids = set()

    curr = now - timedelta(hours=HISTORY_WINDOW)
    # The limit can be simplified as we only need to show what's possible now
    limit_time = now + timedelta(hours=MAX_FORECAST_STEP)

    while curr < limit_time:
        slot_info = get_slot_status(curr)
        valid_slots.append(slot_info)
        processed_ids.add(slot_info["id"])
        curr += timedelta(hours=1)

    disk_state = scan_existing_folders()
    for time_id, entry in disk_state.items():
        if time_id not in processed_ids and entry["ready"]:
            slot_data = {
                "id": time_id,
                "status": "ready",
                "path": f"/{entry['folder']}/",
                "run": entry["run"].strftime("%Y%m%d%H"),
                "step": entry["step"],
            }
            valid_slots.append(slot_data)

    valid_slots.sort(key=lambda x: x["id"])

    return jsonify({"items": valid_slots})


@app.route("/<path:filename>")
def serve_tiles(filename):
    """
    Handles file serving with path rewriting:
    1. Tiles:  /{run_step}/tiles/{z}/{x}/{y}.ktx2 -> /{run_step}/tile_{z}_{x}_{y}.ktx2
    2. Shadow: /{run_step}/shadow.ktx2            -> /{run_step}/shadow.ktx2
    """

    # 1. Tile Route
    # Pattern: run_step/tiles/z/x/y(.sdf).ktx2
    tile_match = re.match(r"^([^/]+)/tiles/(\d+)/(\d+)/(\d+)(\.sdf)?\.ktx2$", filename)
    if tile_match:
        folder, z, x, y, is_sdf = tile_match.groups()
        suffix = ".sdf.ktx2" if is_sdf else ".ktx2"
        real_filename = f"tile_{z}_{x}_{y}{suffix}"

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
    """Queue generation for every available DWD slot that is not yet ready on disk."""
    now = datetime.now(timezone.utc).replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=HISTORY_WINDOW)
    end = now + timedelta(hours=MAX_FORECAST_STEP)

    disk_state = scan_existing_folders()
    already_queued = set()
    queued = 0

    curr = start
    while curr <= end:
        time_id = curr.strftime("%Y%m%d%H")
        best_run, best_step = get_best_run_and_step(curr)
        if best_run:
            best_run_str = best_run.strftime("%Y%m%d%H")
            task_key = (best_run_str, best_step)

            if task_key in already_queued:
                curr += timedelta(hours=1)
                continue

            disk_entry = disk_state.get(time_id)
            if disk_entry and disk_entry["ready"] and disk_entry["run"] == best_run:
                curr += timedelta(hours=1)
                continue

            with processing_lock:
                if task_key in task_progress:
                    curr += timedelta(hours=1)
                    continue

            generation_queue.put((best_run, best_step))
            already_queued.add(task_key)
            queued += 1
            print(f"[AutoBuild] Queued: run {best_run_str} +{best_step}h → target {time_id}")

        curr += timedelta(hours=1)

    print(f"[AutoBuild] Done — {queued} tile set(s) queued")


def archive_old_data():
    """For days older than archive_threshold_days, delete all folders except those
    whose target hour is in archive_keep_hours."""
    threshold = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=config.archive_threshold_days)
    keep_hours = set(config.archive_keep_hours)
    base_dir = os.path.abspath(config.output_dir)

    if not os.path.exists(base_dir):
        return

    removed = 0
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path) or "_" not in name:
            continue
        try:
            r_str, s_str = name.split("_", 1)
            run_dt = datetime.strptime(r_str, "%Y%m%d%H")
            step = int(s_str)
            target_dt = run_dt + timedelta(hours=step)

            if target_dt >= threshold:
                continue

            if target_dt.hour in keep_hours:
                continue

            # Skip if an active task is still working on this folder
            with processing_lock:
                if (r_str, step) in task_progress:
                    continue

            shutil.rmtree(path, ignore_errors=True)
            removed += 1
        except (ValueError, Exception):
            continue

    print(f"[Archive] Removed {removed} old tile set(s) outside keep_hours={sorted(keep_hours)}")


def scheduler_loop():
    """Runs auto_build_all + archive_old_data on startup and daily at auto_build_time."""
    print("[Scheduler] Running initial tasks on startup...")
    auto_build_all()
    archive_old_data()

    while True:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        try:
            hour, minute = map(int, config.auto_build_time.split(":"))
        except ValueError:
            hour, minute = 2, 30

        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)

        sleep_secs = (next_run - now).total_seconds()
        print(f"[Scheduler] Next run at {next_run.strftime('%Y-%m-%d %H:%M')} UTC")
        time.sleep(sleep_secs)

        print("[Scheduler] Running scheduled tasks...")
        auto_build_all()
        archive_old_data()


if __name__ == "__main__":
    output_dir = os.path.abspath(config.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    threading.Thread(target=worker_loop, daemon=True, name="WorkerThread").start()
    threading.Thread(target=scheduler_loop, daemon=True, name="SchedulerThread").start()

    app.run(host="0.0.0.0", port=config.port, debug=False, use_reloader=False)
