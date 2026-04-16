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
import logging
import threading
import config
import log_config
import utils.general as util
import db
import scheduler
import tile_cache
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from waitress import serve

logger = logging.getLogger("server")

VERSION = util.read_version()

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def index():
    return send_from_directory("docs", "index.html")


@app.route("/status", methods=["GET"])
def server_status():
    with scheduler.pending_tasks_lock:
        queued = sorted(f"{run_dt.strftime('%Y%m%d%H')}_{step:03d}" for run_dt, step in scheduler.pending_tasks.values())

    with scheduler.processing_lock:
        active = {
            f"{run_str}_{step:03d}": dict(progress)
            for (run_str, step), progress in scheduler.task_progress.items()
        }

    is_working = bool(queued or active)

    with tile_cache.tile_cache_size_lock:
        cache_size = tile_cache.tile_cache_size

    return jsonify({
        "version": VERSION,
        "status": "working" if is_working else "idle",
        "next_maintenance": scheduler.next_maintenance.strftime("%Y-%m-%dT%H:%M:00Z") if scheduler.next_maintenance else None,
        "active": active,
        "queued": queued,
        "tile_cache": {
            "size": cache_size,
            "max": config.tile_cache_max_size,
        },
    })


@app.route("/available", methods=["GET"])
def list_available():
    disk_state = tile_cache.scan_existing_folders()
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
    return jsonify({"entries": db.log_read_since(since)})


@app.route("/<path:filename>")
def serve_tiles(filename):
    """
    Handles file serving with path rewriting:
    1. Tiles:  /{folder}/tiles/{z}/{x}/{y}.ktx2 -> /{folder}/tile_{z}_{x}_{y}.ktx2
    2. Shadow: /{folder}/shadow.ktx2             -> /{folder}/shadow.ktx2
    """
    tile_match = re.match(r"^([^/]+)/tiles/(\d+)/(\d+)/(\d+)\.ktx2$", filename)
    if tile_match:
        folder, z, x, y = tile_match.groups()
        real_filename = f"tile_{z}_{x}_{y}.ktx2"

        # We send from the specific run_step folder
        return send_from_directory(
            os.path.join(os.path.abspath(config.tile_cache_dir), folder), real_filename
        )
    
    shadow_match = re.match(r"^([^/]+)/shadow\.ktx2$", filename)
    if shadow_match:
        folder = shadow_match.group(1)
        return send_from_directory(
            os.path.join(os.path.abspath(config.tile_cache_dir), folder), "shadow.ktx2"
        )

    return ("Forbidden", 403)


if __name__ == "__main__":
    log_config.setup_logging(log_file=config.log_file)
    log_config.print_logo()
    db.init()
    db.log_append(f"Server started v{VERSION}")
    msg = f" === weBIGeo Cloud Server v{VERSION} started === "
    sep = " " + "=" * (len(msg) - 2) + " "
    logger.info(sep)
    logger.info(msg)
    logger.info(sep)
    cache_dir = os.path.abspath(config.tile_cache_dir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    tile_cache.refresh_tile_cache_size()

    if not config.only_serve:
        for i in range(config.worker_threads):
            threading.Thread(target=scheduler.worker_loop, daemon=True, name=f"WorkerThread-{i}").start()
        threading.Thread(target=scheduler.scheduler_loop, daemon=True, name="SchedulerThread").start()
    else:
        logger.info("only_serve = True -> no background scheduler and worker started")

    logger.info(f"Starting waitress server on http://{config.host}:{config.port}")
    serve(app, host=config.host, port=config.port)
