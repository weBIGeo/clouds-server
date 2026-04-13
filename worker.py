import argparse
import logging
import shutil
import sys
import os
import time
from datetime import datetime
from filelock import FileLock

import log_config

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DOWNLOAD_LOCK  = FileLock(os.path.join(_BASE_DIR, ".download.lock"),      poll_interval=1.0)
_TILE_LOCK      = FileLock(os.path.join(_BASE_DIR, ".tile_processor.lock"), poll_interval=1.0)
_LOD_SHADOW_LOCK = FileLock(os.path.join(_BASE_DIR, ".lod_shadow.lock"),   poll_interval=1.0)

try:
    from dwd_connect import DWDDownloader
    from icon_loader import load_region
    from tile_processor import TileProcessor, TileConfig
    from lod_generator import LODGenerator, LODConfig
    from shadow_map_generator import generate_shadows
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

logger = logging.getLogger("worker")

REGION_CFG = {"lat_min": 46.2, "lat_max": 49.2, "lon_min": 9.4, "lon_max": 17.4}


def run_job(run_time, step, output_dir, max_zoom, keep_gribs=False, skip_lods=False):
    run_str = run_time.strftime("%Y%m%d%H")
    cache_dir = os.path.join(output_dir, "cache")
    job_name = f"{run_str}_{step:03d}"
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Start: {job_name}")

    downloader = DWDDownloader(cache_dir)
    vars_needed = ["hhl", "tke", "clc", "qc", "qi", "qv", "t", "p"]

    with _DOWNLOAD_LOCK:
        t0 = time.time()
        try:
            files_map = downloader.fetch_variables(run_time, step, vars_needed)
            for var in vars_needed:
                if var not in files_map or not files_map[var]:
                    raise FileNotFoundError(f"Missing files for variable: {var}")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            sys.exit(1)
        logger.info(f"Download completed in {time.time() - t0:.2f}s")

        t0 = time.time()
        try:
            cube = load_region(files_map, **REGION_CFG)
        except Exception as e:
            logger.error(f"Load failed: {e}")
            sys.exit(1)
        logger.info(f"Region load completed in {time.time() - t0:.2f}s")

    tile_cfg = TileConfig(
        tile_resolution=256, vertical_layers=64, batch_tiles_x=4, batch_tiles_y=4
    )
    t0 = time.time()
    with _TILE_LOCK:
        processor = TileProcessor(cube, output_dir, config=tile_cfg)
        processor.run_tiled(zoom=max_zoom)
    logger.info(f"Tile processing completed in {time.time() - t0:.2f}s")

    lod_cfg = LODConfig(
        tile_resolution=tile_cfg.tile_resolution,
        vertical_layers=tile_cfg.vertical_layers,
    )
    lod_start_zoom = max_zoom if skip_lods else 4
    with _LOD_SHADOW_LOCK:
        t0 = time.time()
        lod_gen = LODGenerator(
            output_dir, config=lod_cfg, max_zoom=max_zoom, start_zoom=lod_start_zoom
        )
        lod_gen.run()
        logger.info(f"LOD generation completed in {time.time() - t0:.2f}s")

        if skip_lods:
            logger.warning("Cannot generate shadow due to skipped LOD generation")
        else:
            t0 = time.time()
            shadow_out = os.path.join(output_dir, "shadow.ktx2")
            generate_shadows(output_dir, shadow_out, REGION_CFG, lod_cfg)
            logger.info(f"Shadow generation completed in {time.time() - t0:.2f}s")

    if not keep_gribs:
        logger.debug("Deleting GRIBs cache")
        try:
            shutil.rmtree(cache_dir)
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")

    logger.info(f"Done: {job_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-zoom", type=int, default=10)
    parser.add_argument("--keep-gribs", action="store_true")
    parser.add_argument("--skip-lods", action="store_true")

    args = parser.parse_args()

    log_config.setup_logging()
    try:
        dt = datetime.strptime(args.run, "%Y%m%d%H")
        run_job(dt, args.step, args.out, args.max_zoom, args.keep_gribs, args.skip_lods)
    except Exception as e:
        logger.critical(f"Crash: {repr(e)}")
        sys.exit(1)
