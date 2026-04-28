#############################################################################
# weBIGeo Clouds
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
import config
import db
import scheduler
import utils.general as util
from flask import Blueprint, jsonify, request, send_from_directory

bp = Blueprint("v2", __name__, url_prefix="/v2")

VERSION = util.read_version()


@bp.route("/status")
def status():
    with scheduler.processing_lock:
        active = {
            f"{run_str}_{step:03d}": dict(progress)
            for (run_str, step), progress in scheduler.task_progress.items()
        }
    return jsonify({
        "version": VERSION,
        "status": "working" if active else "idle",
        "next_maintenance": scheduler.next_maintenance.strftime("%Y-%m-%dT%H:%M:00Z") if scheduler.next_maintenance else None,
        "active": active,
        "tilesets": {
            "size": db.tileset_get_size(),
            "max": config.tilesets_max_size,
        },
    })


@bp.route("/tilesets")
def list_tilesets():
    status_filter = request.args.get("status") or None
    rows = db.tileset_get_all(status=status_filter)
    items = [
        {
            "id": r["target_str"],
            "folder": r["folder"],
            "status": r["status"],
            "size": r["size"],
            "queued_at": r["queued_at"],
            "completed_at": r["completed_at"],
            "processing_time": r["processing_time"],
        }
        for r in rows
    ]
    items.sort(key=lambda x: x["id"])
    return jsonify({"items": items})


@bp.route("/log")
def get_log():
    try:
        since = int(request.args.get("since", 7 * 24 * 3600))
    except (ValueError, TypeError):
        since = 7 * 24 * 3600
    return jsonify({"entries": db.log_read_since(since)})


@bp.route("/<path:filename>")
def serve_tiles(filename):
    tile_match = re.match(r"^([^/]+)/tiles/(\d+)/(\d+)/(\d+)\.ktx2$", filename)
    if tile_match:
        folder, z, x, y = tile_match.groups()
        return send_from_directory(
            os.path.join(os.path.abspath(config.tileset_cache_dir), folder),
            f"tile_{z}_{x}_{y}.ktx2",
        )
    shadow_match = re.match(r"^([^/]+)/shadow\.ktx2$", filename)
    if shadow_match:
        folder = shadow_match.group(1)
        return send_from_directory(
            os.path.join(os.path.abspath(config.tileset_cache_dir), folder),
            "shadow.ktx2",
        )
    return ("Forbidden", 403)
