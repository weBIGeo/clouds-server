import os
import sys
import shutil
import subprocess
import threading
import queue
import argparse
import re
import urllib.request
import urllib.error
from datetime import datetime, timedelta
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

SERVER_ARGS = {}

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
    now = datetime.utcnow()

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
            DWD_RUN_CACHE[run_str] = {"steps": set(), "status": "fail", "ts": datetime.utcnow()}
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
            DWD_RUN_CACHE[run_str] = {"steps": steps, "status": "success", "ts": datetime.utcnow()}
        else:
            DWD_RUN_CACHE[run_str] = {"steps": set(), "status": "fail", "ts": datetime.utcnow()}

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
    now = datetime.utcnow()
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
        BEST_RUN_CACHE[time_id] = {"run": best[0], "step": best[1], "ts": datetime.utcnow()}

    return best


def get_folder_path(run_time, step):
    folder_name = f"{run_time.strftime('%Y%m%d%H')}_{step:03d}"
    return os.path.join(SERVER_ARGS["dir"], folder_name), folder_name


def scan_existing_folders():
    results = {}
    base_dir = SERVER_ARGS["dir"]
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
    if not SERVER_ARGS["clean"]:
        return

    cutoff = datetime.utcnow() - timedelta(hours=HISTORY_WINDOW)

    folders = []
    base_dir = SERVER_ARGS["dir"]
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
                    progress = {
                        "stage": stage,
                        "detail": detail,
                        "percent": int(percent_str),
                    }
                    with processing_lock:
                        if task_key in task_progress:
                            task_progress[task_key].update(progress)
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
        output_dir = os.path.join(SERVER_ARGS["dir"], folder_name)
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

        if(SERVER_ARGS["keep-gribs"]):
            cmd.append("--keep-gribs")

        with processing_lock:
            task_progress[task_key] = {
                "status": "pending",
                "stage": "initializing",
                "detail": "starting worker process",
                "percent": 0,
            }

        print(f"[Server] Processing: {run_str} +{step}h")
        try:
            # Use Popen for non-blocking execution
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            # Start a thread to read the output in real-time
            reader_thread = threading.Thread(
                target=worker_output_reader,
                args=(process, task_key, log_path),
                daemon=True,
            )
            reader_thread.start()

            # Wait for the process to complete
            process.wait()

            # Optional: Check return code
            if process.returncode != 0:
                print(
                    f"[Server] Worker for {task_key} failed. See {log_path} for details."
                )

        except Exception as e:
            print(f"[Server] Error launching worker: {e}")
        finally:
            with processing_lock:
                task_progress.pop(task_key, None)

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
    min_time = datetime.utcnow() - timedelta(hours=HISTORY_WINDOW)
    max_time = datetime.utcnow() + timedelta(hours=MAX_FORECAST_STEP)
    is_in_window = min_time <= target_time <= max_time

    if is_in_window:
        status = get_slot_status(target_time)
        return jsonify(status)

    if SERVER_ARGS["clean"]:
        return (
            jsonify({"status": "error", "message": "Time is outside history window"}),
            404,
        )

    # If NOT in clean mode, check the disk for historical data.
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

    if SERVER_ARGS["readonly"]:
        return (
            jsonify({"status": "error", "message": "Server is in read-only mode"}),
            403,
        )

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
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
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

    if not SERVER_ARGS["clean"]:
        disk_state = scan_existing_folders()

        for time_id, entry in disk_state.items():
            # Add only if it wasn't already processed and is marked as ready
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
            os.path.join(SERVER_ARGS["dir"], folder), real_filename
        )

    # 2. Shadow Route
    # Pattern: run_step/shadow.ktx2
    shadow_match = re.match(r"^([^/]+)/shadow\.ktx2$", filename)
    if shadow_match:
        folder = shadow_match.group(1)
        return send_from_directory(
            os.path.join(SERVER_ARGS["dir"], folder), "shadow.ktx2"
        )

    return ("Forbidden", 403)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="tiles_output")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--readonly", action="store_true")
    parser.add_argument("--keep-gribs", action="store_true")

    args = parser.parse_args()
    SERVER_ARGS["dir"] = os.path.abspath(args.dir)
    SERVER_ARGS["clean"] = args.clean
    SERVER_ARGS["readonly"] = args.readonly
    SERVER_ARGS["keep-gribs"] = args.keep_gribs

    if not os.path.exists(SERVER_ARGS["dir"]):
        os.makedirs(SERVER_ARGS["dir"])

    if not args.readonly:
        t = threading.Thread(target=worker_loop, daemon=True, name="WorkerThread")
        t.start()

    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)
