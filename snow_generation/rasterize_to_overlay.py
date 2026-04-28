#############################################################################
# weBIGeo Snow
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
"""
Convert SNOWGRID GeoJSON data to webigeo image overlays (PNG + AABB txt).

Produces one PNG per variable (mode 1 float encoding) and one shared AABB
file, since all variables share the same spatial grid.

Output files:
    data/overlay/<date>_snow_depth.png  -- float-encoded RGBA texture
    data/overlay/<date>_swe_tot.png     -- float-encoded RGBA texture
    data/overlay/<date>_aabb.txt        -- shared EPSG:3857 bounds

webigeo settings when loading:
    mode                    = 1
    float_decoding_lower_bound = 0.0
    float_decoding_upper_bound = <printed max value per variable>
"""

import logging
import os
import sys

import numpy as np
from PIL import Image
from pyproj import Transformer
from scipy.interpolate import griddata

sys.path.insert(0, os.path.dirname(__file__))
from geosphere_connect import SNOWGRID_PARAMETERS, TARGET_DATE, fetch_snowgrid

logger = logging.getLogger("rasterize_to_overlay")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "overlay")

# Raster cell size in EPSG:3857 meters.
# Source grid is 1 km in EPSG:3416; at Austria's latitude (~47.5°N) the Web
# Mercator scale factor is ~1.49, so 1500 m covers one source cell without gaps.
EPSG3857_RES = 1000 #1500

INTERPOLATION_METHOD = "linear"  # "nearest" | "linear" | "cubic"
TREAT_NAN_AS_ZERO = True         # replace NaN (outside hull / missing data) with 0

# The shader decodes RGBA → u32 → maps to this range before applying the
# per-variable lower/upper bounds (see encoder.wgsl: U32_ENCODING_RANGE_VALIDATION).
_ENCODE_RANGE_MIN = -10000.0
_ENCODE_RANGE_MAX = 10000.0


def _encode_float_grid(grid: np.ndarray) -> np.ndarray:
    """
    Encode a 2-D float array to RGBA uint8 matching webigeo mode-1 decoding:
        decode_rgba_to_normalized_value → u32_to_range([-10000, 10000])

    NaN cells → (0, 0, 0, 0): decoded = -10000, which falls below any valid
    lower_bound and is discarded by the shader's `encoded_float_value > 0.0` guard.
    """
    span = _ENCODE_RANGE_MAX - _ENCODE_RANGE_MIN
    normalized = np.where(
        ~np.isnan(grid),
        np.clip((grid - _ENCODE_RANGE_MIN) / span, 0.0, 1.0),
        0.0,
    )
    packed = (normalized * 0xFFFFFFFF).astype(np.uint64)  # u64 avoids overflow

    rgba = np.zeros((*grid.shape, 4), dtype=np.uint8)
    valid = ~np.isnan(grid)
    rgba[valid, 0] = ((packed[valid] >> 24) & 0xFF).astype(np.uint8)
    rgba[valid, 1] = ((packed[valid] >> 16) & 0xFF).astype(np.uint8)
    rgba[valid, 2] = ((packed[valid] >> 8) & 0xFF).astype(np.uint8)
    rgba[valid, 3] = (packed[valid] & 0xFF).astype(np.uint8)
    return rgba


def rasterize_all(date_str: str) -> None:
    """
    Fetch SNOWGRID data for *date_str*, rasterize all variables, and write
    float-encoded PNGs plus a single shared AABB file.
    """
    data, _ = fetch_snowgrid(date_str)
    features = data.get("features", [])
    if not features:
        raise ValueError(f"No features in SNOWGRID response for {date_str}")

    # GeoJSON from GeoSphere is WGS84; convert once to EPSG:3857 for webigeo.
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    lons = np.array([f["geometry"]["coordinates"][0] for f in features], dtype=np.float64)
    lats = np.array([f["geometry"]["coordinates"][1] for f in features], dtype=np.float64)
    xs, ys = transformer.transform(lons, lats)

    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))

    nx = int((x_max - x_min) / EPSG3857_RES) + 1
    ny = int((y_max - y_min) / EPSG3857_RES) + 1
    logger.info(f"Grid: {nx}×{ny} px  AABB: [{x_min:.0f}, {y_min:.0f}] → [{x_max:.0f}, {y_max:.0f}]")

    # Output pixel-centre coordinates; row 0 = northernmost (image top).
    xs_out, ys_out = np.meshgrid(
        np.linspace(x_min, x_max, nx),
        np.linspace(y_max, y_min, ny),
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Shared AABB file (webigeo format: 4 lines, one float each).
    aabb_path = os.path.join(OUTPUT_DIR, f"{date_str}_aabb.txt")
    with open(aabb_path, "w") as f:
        f.write(f"{x_min}\n{y_min}\n{x_max}\n{y_max}\n")
    logger.info(f"AABB: {aabb_path}")

    # Points overview texture: turquoise = has any valid value, red = all NaN.
    all_vals = np.array(
        [[f.get("properties", {}).get("parameters", {}).get(v, {}).get("data", [None])[0]
          for v in SNOWGRID_PARAMETERS] for f in features],
        dtype=np.float32,
    )
    any_valid = ~np.all(np.isnan(all_vals), axis=1)
    ov_nx, ov_ny = nx * 4, ny * 4
    ov_cols = np.clip(((xs - x_min) / (x_max - x_min) * (ov_nx - 1)).astype(int), 0, ov_nx - 1)
    ov_rows = np.clip(((y_max - ys) / (y_max - y_min) * (ov_ny - 1)).astype(int), 0, ov_ny - 1)
    ov_img = np.zeros((ov_ny, ov_nx, 4), dtype=np.uint8)
    ov_img[ov_rows, ov_cols] = [255, 0, 0, 255]                                   # red (NaN)
    ov_img[ov_rows[any_valid], ov_cols[any_valid]] = [0, 206, 209, 255]            # turquoise (valid)
    ov_path = os.path.join(OUTPUT_DIR, f"{date_str}_points.png")
    Image.fromarray(ov_img, "RGBA").save(ov_path)
    logger.info(f"Points overview: {ov_path}  ({ov_nx}×{ov_ny} px, {any_valid.sum()}/{len(features)} valid)")

    for variable in SNOWGRID_PARAMETERS:
        raw = [
            f.get("properties", {}).get("parameters", {}).get(variable, {}).get("data", [None])[0]
            for f in features
        ]
        vals = np.array([v if v is not None else np.nan for v in raw], dtype=np.float32)

        valid = ~np.isnan(vals)
        if not np.any(valid):
            grid = np.full((ny, nx), np.nan, dtype=np.float32)
        else:
            grid = griddata(
                (xs[valid], ys[valid]),
                vals[valid],
                (xs_out, ys_out),
                method=INTERPOLATION_METHOD,
            ).astype(np.float32)

        if TREAT_NAN_AS_ZERO:
            grid = np.nan_to_num(grid, nan=0.0)

        max_val = float(np.nanmax(grid)) if np.any(valid) else 1.0
        rgba = _encode_float_grid(grid)

        png_path = os.path.join(OUTPUT_DIR, f"{date_str}_{variable}.png")
        Image.fromarray(rgba, "RGBA").save(png_path)

        # Point texture: 4× size, valid source points scattered with encoded values.
        pt_nx, pt_ny = nx * 4, ny * 4
        pt_grid = np.full((pt_ny, pt_nx), np.nan, dtype=np.float32)
        pt_cols = np.clip(((xs[valid] - x_min) / (x_max - x_min) * (pt_nx - 1)).astype(int), 0, pt_nx - 1)
        pt_rows = np.clip(((y_max - ys[valid]) / (y_max - y_min) * (pt_ny - 1)).astype(int), 0, pt_ny - 1)
        pt_grid[pt_rows, pt_cols] = vals[valid]
        pt_path = os.path.join(OUTPUT_DIR, f"{date_str}_{variable}_points.png")
        Image.fromarray(_encode_float_grid(pt_grid), "RGBA").save(pt_path)

        logger.info(
            f"[{variable}] PNG: {png_path}  "
            f"max={max_val:.4f}  → set float_decoding_upper_bound={max_val:.4f}  "
            f"points: {pt_path}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    try:
        rasterize_all(TARGET_DATE)
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)
