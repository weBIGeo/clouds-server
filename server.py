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

import logging
import threading
import config
import log_config
import utils.general as util
import db
import scheduler
import tilesets
import routes_v1
import routes_v2
from flask import Flask, send_from_directory
from flask_cors import CORS
from waitress import serve

logger = logging.getLogger("server")

VERSION = util.read_version()

app = Flask(__name__)
CORS(app)
app.register_blueprint(routes_v1.bp)
app.register_blueprint(routes_v2.bp)


@app.route("/", methods=["GET"])
def index():
    return send_from_directory("docs", "index.html")


# Unversioned aliases — delegates to routes_v2 handlers

@app.route("/status", methods=["GET"])
def server_status():
    return routes_v2.status()


@app.route("/tilesets", methods=["GET"])
def list_tilesets():
    return routes_v2.list_tilesets()


@app.route("/log", methods=["GET"])
def get_public_log():
    return routes_v2.get_log()


@app.route("/<path:filename>")
def serve_tiles(filename):
    return routes_v2.serve_tiles(filename)


if __name__ == "__main__":
    log_config.setup_logging(log_file=config.log_file)
    log_config.print_logo()
    db.init(config.db_path)
    db.log_append(f"Server started v{VERSION}")
    msg = f" === weBIGeo Cloud Server v{VERSION} started === "
    sep = " " + "=" * (len(msg) - 2) + " "
    logger.info(sep)
    logger.info(msg)
    logger.info(sep)
    tilesets.sync_from_disk()
    if db.tileset_count_pending() > 0:
        scheduler.pending_tasks_available.set()

    if not config.only_serve:
        for i in range(config.worker_threads):
            threading.Thread(target=scheduler.worker_loop, daemon=True, name=f"WorkerThread-{i}").start()
        threading.Thread(target=scheduler.scheduler_loop, daemon=True, name="SchedulerThread").start()
    else:
        logger.info("only_serve = True -> no background scheduler and worker started")

    logger.info(f"Starting waitress server on http://{config.host}:{config.port}")
    serve(app, host=config.host, port=config.port)
