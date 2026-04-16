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

import json
import os
import sys
import shutil
import subprocess
import threading
import time
import logging
import config
import db
from datetime import datetime, timedelta, timezone
from tile_cache import (
    scan_existing_folders,
    get_best_run_and_step,
    get_scheme_rule,
    compute_folder_size,
)

logger = logging.getLogger("scheduler")

pending_tasks_ready = threading.Event()
processing_lock = threading.Lock()
task_progress = {}
next_maintenance: datetime | None = None


def worker_output_reader(process, task_key, log_file_path):
    """Reads stdout from a worker process and updates the shared progress dict."""
    run_start = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\nRun attempt: {run_start}\n{'=' * 60}\n")
        f.flush()
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

        task = db.tile_claim_pending()
        if task is None:
            if db.tile_count_pending() == 0:
                pending_tasks_ready.clear()
            continue

        folder_name = task["folder"]
        run_str = task["run_str"]
        step = task["step"]
        run_dt = datetime.strptime(run_str, "%Y%m%d%H")
        task_key = (run_str, step)
        maintenance_id = task.get("maintenance_id")

        current_cache_size = db.tile_get_cache_size()
        if current_cache_size >= config.tile_cache_max_size:
            logger.warning(
                f"Tile cache size ({current_cache_size / 1e9:.1f} GB) exceeds limit "
                f"({config.tile_cache_max_size / 1e9:.0f} GB), skipping {run_str}+{step}h"
            )
            db.tile_set_status(folder_name, "pending")
            pending_tasks_ready.clear()
            continue

        output_dir = os.path.join(os.path.abspath(config.tile_cache_dir), folder_name)
        os.makedirs(output_dir, exist_ok=True)

        invalid_path = os.path.join(output_dir, "invalid")
        open(invalid_path, "w").close()

        log_path = os.path.join(output_dir, "latest.log")

        cmd = [
            sys.executable, "-m", "cloud_generation.worker",
            "--run", run_str,
            "--step", str(step),
            "--out", output_dir,
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

        queue_depth = db.tile_count_pending()
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
                db.tile_set_status(folder_name, "failed")
            else:
                os.remove(invalid_path)
                elapsed = time.monotonic() - start_time
                size = compute_folder_size(output_dir)
                db.tile_set_ready(folder_name, size, elapsed)
                logger.info(f"Done: run {run_str} +{step}h ({elapsed:.1f}s, {size / 1e6:.1f} MB)")
                success = True

        except Exception as e:
            logger.error(f"Error launching worker: {e}")
            db.tile_set_status(folder_name, "failed")
        finally:
            with processing_lock:
                task_progress.pop(task_key, None)

        # Remove any stale folders for the same target time
        if success:
            target_dt = run_dt + timedelta(hours=step)
            base_dir = os.path.abspath(config.tile_cache_dir)
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
                        db.tile_delete(name)
                        logger.debug(f"Removed stale folder: {name}")
                        if maintenance_id is not None:
                            db.maintenance_add_renewed(maintenance_id, name)
                except ValueError:
                    continue

        _check_maintenance_completion()


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

    # Build lookup of target_str -> DB row for already-known targets
    db_state = {r["target_str"]: r for r in db.tile_get_all()}

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

        db_entry = db_state.get(time_id)
        folder = f"{best_run.strftime('%Y%m%d%H')}_{best_step:03d}"
        run_str = best_run.strftime("%Y%m%d%H")

        if db_entry:
            if db_entry["status"] == "fetching":
                continue  # currently running, don't interfere
            if db_entry["status"] in ("ready", "pending") and db_entry["folder"] == folder:
                continue  # already at best run
            if db_entry["status"] == "pending" and db_entry["folder"] != folder:
                # Newer run available — delete the stale pending entry and queue the new one
                db.tile_delete(db_entry["folder"])
                logger.debug(f"AutoBuild: replaced stale pending {db_entry['folder']} with {folder}")
            elif db_entry["status"] == "failed" and db_entry["folder"] == folder:
                # Re-attempt same failed run on next maintenance
                db.tile_set_status(folder, "pending")
                queued.append((folder, task_key))
                added += 1
                logger.debug(f"AutoBuild: retrying failed {folder} (target {time_id})")
                continue

        inserted = db.tile_upsert(folder, run_str, best_step, time_id)
        if inserted:
            queued.append((folder, task_key))
            added += 1
            logger.debug(f"AutoBuild: {folder} (target {time_id})")

    logger.info(f"AutoBuild done: {added} added")

    # Signal workers if there are pending tasks (including previously queued ones)
    if db.tile_count_pending() > 0:
        pending_tasks_ready.set()

    return queued


def purge_old_data():
    """Remove invalid dirs and apply the tile_retention_policy rules."""
    base_dir = os.path.abspath(config.tile_cache_dir)
    if not os.path.exists(base_dir):
        return []

    # Build set of folders currently being fetched or failed in DB — skip purging these
    fetching_folders = {r["folder"] for r in db.tile_get_all(status="fetching")}
    failed_folders = {r["folder"] for r in db.tile_get_all(status="failed")}

    purged: list[str] = []
    for entry in scan_existing_folders().values():
        path = os.path.join(base_dir, entry["folder"])

        with processing_lock:
            if (entry["run"].strftime("%Y%m%d%H"), entry["step"]) in task_progress:
                continue

        if entry["folder"] in fetching_folders or entry["folder"] in failed_folders:
            continue

        if not entry["ready"]:
            logger.debug(f"Purge: {entry['folder']} (invalid/incomplete)")
            shutil.rmtree(path, ignore_errors=True)
            db.tile_delete(entry["folder"])
            purged.append(entry["folder"])
            continue

        target_dt = entry["run"] + timedelta(hours=entry["step"])
        rule = get_scheme_rule(target_dt)

        if rule is None:
            continue  # beyond all rules, leave untouched

        if rule["mode"] != "fetch_only" and target_dt.hour not in rule["hours"]:
            logger.debug(f"Purge: {entry['folder']} (hour {target_dt.hour} not in retention policy hours {rule['hours']})")
            shutil.rmtree(path, ignore_errors=True)
            db.tile_delete(entry["folder"])
            purged.append(entry["folder"])

    logger.info(f"Purge: removed {len(purged)} tile set(s)")
    return purged


def _check_maintenance_completion():
    """Called by worker_loop after each task finishes.
    Completes any maintenance record whose tiles are all done."""
    for row in db.maintenance_get_incomplete():
        if db.tile_count_active_for_maintenance(row["id"]) > 0:
            continue
        completed = db.maintenance_complete(row["id"])
        added   = json.loads(completed["added"])
        purged  = json.loads(completed["purged"])
        renewed = json.loads(completed["renewed"])
        started = datetime.strptime(completed["started_at"], "%Y-%m-%dT%H:%M:%SZ")
        done    = datetime.strptime(completed["completed_at"], "%Y-%m-%dT%H:%M:%SZ")
        elapsed = (done - started).total_seconds() / 60
        added_str   = ", ".join(added)   or "none"
        purged_str  = ", ".join(purged)  or "none"
        renewed_str = ", ".join(renewed) or "none"
        summary = f"{completed['label']} Maintenance completed after {elapsed:.1f} min"
        logger.info(summary)
        db.log_append(
            f"{summary}:\n"
            f" - {len(added)} added ({added_str})\n"
            f" - {len(purged)} purged ({purged_str})\n"
            f" - {len(renewed)} renewed ({renewed_str})"
        )


def _run_maintenance(name=None):
    label = name if name else datetime.now(timezone.utc).strftime("%H:%M UTC")
    logger.info("Scheduler: running maintenance")
    purged_folders = purge_old_data()
    queued = auto_build_all()
    if queued:
        mid = db.maintenance_create(label, [f for f, _ in queued], purged_folders)
        for folder, _ in queued:
            db.tile_set_maintenance(folder, mid)
    else:
        purged_str = ", ".join(purged_folders) or "none"
        summary = f"{label} Maintenance completed after 0.0 min"
        logger.info(summary)
        db.log_append(
            f"{summary}:\n"
            f" - 0 added (none)\n"
            f" - {len(purged_folders)} purged ({purged_str})\n"
            f" - 0 renewed (none)"
        )


def scheduler_loop():
    """Runs _run_maintenance on startup and at each configured auto_build_time."""
    global next_maintenance
    _run_maintenance(name="Startup")

    while True:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        times = [now.replace(hour=int(e[:2]), minute=int(e[3:]), second=0, microsecond=0)
                 for e in config.auto_build_time]
        next_maintenance = next((t for t in times if t > now), times[0] + timedelta(days=1))
        logger.info(f"Scheduler: next maintenance at {next_maintenance.strftime('%Y-%m-%d %H:%M')} UTC")
        time.sleep((next_maintenance - now).total_seconds())
        _run_maintenance()
