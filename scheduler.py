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
import time
import logging
import config
import db
from datetime import datetime, timedelta, timezone
from tile_cache import (
    get_folder_path,
    scan_existing_folders,
    refresh_tile_cache_size,
    tile_cache_size_lock,
    get_best_run_and_step,
    get_scheme_rule,
    _scheme_fetch_window,
)
import tile_cache

logger = logging.getLogger("scheduler")

pending_tasks = {}  # target_time_id -> (run_dt, step)
pending_tasks_lock = threading.Lock()
pending_tasks_ready = threading.Event()
active_tasks = set()
processing_lock = threading.Lock()
task_progress = {}
next_maintenance: datetime | None = None
maintenance_batches: list[dict] = []
maintenance_batches_lock = threading.Lock()


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

        with tile_cache_size_lock:
            current_cache_size = tile_cache.tile_cache_size
        if current_cache_size >= config.tile_cache_max_size:
            logger.warning(
                f"Tile cache size ({current_cache_size / 1e9:.1f} GB) exceeds limit "
                f"({config.tile_cache_max_size / 1e9:.0f} GB), skipping {run_str}+{step}h"
            )
            continue

        folder_name = f"{run_str}_{step:03d}"
        output_dir = os.path.join(os.path.abspath(config.tile_cache_dir), folder_name)
        os.makedirs(output_dir, exist_ok=True)

        invalid_path = os.path.join(output_dir, "invalid")
        open(invalid_path, "w").close()

        log_path = os.path.join(output_dir, "latest.log")

        cmd = [
            sys.executable, "-m", "cloud_generation.worker",
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

        refresh_tile_cache_size()
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
    base_dir = os.path.abspath(config.tile_cache_dir)
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
        elapsed = (time.monotonic() - batch["start_time"]) / 60
        added_str = ", ".join(batch["folder_names"]) or "none"
        purged_str = ", ".join(batch["purged"]) or "none"
        renewed_str = ", ".join(batch["renewed"]) or "none"
        summary = f"{batch['name']} Maintenance completed after {elapsed:.1f} min"
        logger.info(summary)
        db.log_append(f"{summary}:\n - {len(batch['folder_names'])} added ({added_str})\n - {len(batch['purged'])} purged ({purged_str})\n - {len(batch['renewed'])} renewed ({renewed_str})")


def _run_maintenance(name=None):
    start_time = time.monotonic()
    label = name if name else datetime.now(timezone.utc).strftime("%H:%M UTC")
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
                "name":         label,
            })
    else:
        elapsed    = (time.monotonic() - start_time) / 60
        purged_str = ", ".join(purged_folders) or "none"
        summary    = f"{label} Maintenance completed after {elapsed:.1f} min"
        logger.info(summary)
        db.log_append(f"{summary}:\n - 0 added (none)\n - {len(purged_folders)} purged ({purged_str})\n - 0 renewed (none)")


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
