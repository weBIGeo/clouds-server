import logging
import math
import os
import argparse
import numpy as np
import mercantile
from dataclasses import dataclass
from numba import jit, prange
from io_ktx import Ktx2
from bc4 import BC4Compressor
from scipy.ndimage import gaussian_filter
from util import MAX_ALTITUDE, report_progress

logger = logging.getLogger("shadows")


@dataclass
class ShadowConfig:
    zoom_level: int = 7
    tile_resolution: int = 256
    vertical_layers: int = 31
    shadow_threshold: float = 2.0
    max_density: float = 1.0
    esm_constant: float = 4.0
    blur_sigma: float = 1.0


@jit(nopython=True, fastmath=True, parallel=True)
def calculate_tile_shadows(density_vol, threshold):
    """
    Raymarches top-down through the density volume.
    Returns a 2D height map (0.0 to 1.0) where optical depth > threshold.
    """
    nz, ny, nx = density_vol.shape
    out = np.zeros((ny, nx), dtype=np.float32)

    # Iterate over pixels
    for y in prange(ny):
        for x in range(nx):
            optical_depth = 0.0
            hit_height = 0.0

            # Top-down integration (z=0 is top)
            for z in range(nz):
                val = density_vol[z, y, x]
                optical_depth += val

                if optical_depth > threshold:
                    # Normalized height: 1.0 is top, 0.0 is bottom
                    hit_height = 1.0 - (z / nz)
                    break

            out[y, x] = hit_height

    return out


@jit(nopython=True, fastmath=True, parallel=True)
def calculate_tile_shadows_esm(density_vol, threshold, esm_c):
    """
    Calculates the ESM value for each pixel.
    Instead of height, it returns exp(C * height).
    """
    nz, ny, nx = density_vol.shape
    out = np.zeros((ny, nx), dtype=np.float32)

    for y in prange(ny):
        for x in range(nx):
            optical_depth = 0.0
            hit_height = 0.0  # Default to ground level

            for z in range(nz):
                val = density_vol[z, y, x]
                optical_depth += val

                if optical_depth > threshold:
                    hit_height = 1.0 - (z / nz)
                    break

            out[y, x] = np.exp(esm_c * hit_height)

    return out


class ShadowMapGenerator:
    def __init__(self, data_dir: str, config: ShadowConfig = None):
        self.data_dir = data_dir
        self.config = config or ShadowConfig()
        self.decompressor = BC4Compressor()

    def _load_and_process_tile(self, x_tms: int, y_tms: int) -> np.ndarray:
        filename = f"tile_{self.config.zoom_level}_{x_tms}_{y_tms}.ktx2"
        path = os.path.join(self.data_dir, filename)

        if not os.path.exists(path):
            return None

        try:
            mips, meta = Ktx2.load(path)
            raw_bytes = mips[0]

            base_w, base_h, base_d = meta["_ktx_block_dimensions"]
            vol = self.decompressor.decompress(raw_bytes, base_w, base_h, base_d)

            # Scale to physical units
            if self.config.max_density != 1.0:
                vol *= self.config.max_density

            # Flip Y (KTX texture coords vs Data array) to match processing logic
            vol = np.flip(vol, axis=1)

            # return calculate_tile_shadows(vol, self.config.shadow_threshold)
            return calculate_tile_shadows_esm(
                vol, self.config.shadow_threshold, self.config.esm_constant
            ).astype(np.float16)

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            return None

    def generate(self, region: dict, output_path: str):
        report_progress("shadows", "", 0)
        z = self.config.zoom_level

        # Calculate tile range
        tiles = list(
            mercantile.tiles(
                region["lon_min"],
                region["lat_min"],
                region["lon_max"],
                region["lat_max"],
                z,
            )
        )

        if not tiles:
            logger.warning("No tiles found for the specified region")
            return

        min_x = min(t.x for t in tiles)
        max_x = max(t.x for t in tiles)
        min_y = min(t.y for t in tiles)
        max_y = max(t.y for t in tiles)

        width_tiles = max_x - min_x + 1
        height_tiles = max_y - min_y + 1
        res = self.config.tile_resolution

        # Initialize canvas
        full_h = height_tiles * res
        full_w = width_tiles * res
        canvas = np.zeros((full_h, full_w), dtype=np.float32)

        report_progress("shadows", f"Stitching {width_tiles}x{height_tiles} tiles...", 25)

        # Process and stitch
        for t in tiles:
            # Convert Google XYZ to TMS for file lookup
            y_tms = (1 << z) - 1 - t.y

            shadow_tile = self._load_and_process_tile(t.x, y_tms)

            if shadow_tile is not None:
                # Calculate canvas offsets
                # Mercantile/Google Y increases South, matches Numpy array
                off_x = (t.x - min_x) * res
                off_y = (t.y - min_y) * res
                canvas[off_y : off_y + res, off_x : off_x + res] = shadow_tile

        if self.config.blur_sigma > 0:
            report_progress("shadows", f"Applying Gaussian blur with sigma={self.config.blur_sigma}...", 50)
            canvas = gaussian_filter(canvas, sigma=self.config.blur_sigma)

        # Crop to precise lat/lon
        report_progress("shadows", f"Saving to {output_path}", 75)
        self._crop_and_save(canvas, region, min_x, max_x, min_y, max_y, output_path)

        report_progress("shadows", "Saving output", 100)

    def _crop_and_save(self, canvas, region, min_x, max_x, min_y, max_y, path):

        full_h, full_w = canvas.shape

        # Get meter bounds of the stitched canvas
        ul_bounds = mercantile.xy_bounds(
            mercantile.Tile(min_x, min_y, self.config.zoom_level)
        )
        lr_bounds = mercantile.xy_bounds(
            mercantile.Tile(max_x, max_y, self.config.zoom_level)
        )

        canvas_min_xm = ul_bounds.left
        canvas_max_ym = ul_bounds.top
        canvas_width_m = lr_bounds.right - ul_bounds.left
        canvas_height_m = ul_bounds.top - lr_bounds.bottom

        # Project target crop region to meters
        crop_ul_xm, crop_ul_ym = mercantile.xy(region["lon_min"], region["lat_max"])
        crop_lr_xm, crop_lr_ym = mercantile.xy(region["lon_max"], region["lat_min"])

        # Map meters to pixels
        def get_pix_x(xm, func):
            pct = (xm - canvas_min_xm) / canvas_width_m
            return int(np.clip(func(pct * full_w), 0, full_w))

        def get_pix_y(ym, func):
            # Y pixel 0 is Top (Max Y meters)
            pct = (canvas_max_ym - ym) / canvas_height_m
            return int(np.clip(func(pct * full_h), 0, full_h))

        x0 = get_pix_x(crop_ul_xm, math.floor)
        x1 = get_pix_x(crop_lr_xm, math.ceil)
        y0 = get_pix_y(crop_ul_ym, math.floor)
        y1 = get_pix_y(crop_lr_ym, math.ceil)

        cropped = canvas[y0:y1, x0:x1]

        # --- Bounding Box Calculation ---
        # Convert the final pixel crop boundaries (x0, y0, x1, y1) back to world meters
        # This ensures the bounding box perfectly aligns with the output texture pixels.
        def get_meters(px, py):
            mx = canvas_min_xm + (px / full_w) * canvas_width_m
            my = canvas_max_ym - (py / full_h) * canvas_height_m
            return mx, my

        bbox_min_x, bbox_max_y = get_meters(x0, y0)  # Top-left corner
        bbox_max_x, bbox_min_y = get_meters(x1, y1)

        final_data = cropped[:, :, np.newaxis]

        logger.debug(
            f"Shadow map bounding box (EPSG:3857): "
            f"min=({bbox_min_x:.8f}, {bbox_min_y:.8f}, 0.0) "
            f"max=({bbox_max_x:.8f}, {bbox_max_y:.8f}, {MAX_ALTITUDE:.1f})"
        )

        Ktx2.save(
            data=final_data,
            path=path,
            target_format="R16_SFLOAT",
            compress=True,
            dimension=2,
        )


def generate_shadows(data_dir, output_path, crop_region, lod_config=None):
    """Entry point for pipeline integration."""
    config = ShadowConfig()
    if lod_config:
        config.vertical_layers = lod_config.vertical_layers
        config.max_density = lod_config.max_density_value

    generator = ShadowMapGenerator(data_dir, config)
    generator.generate(crop_region, output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate 2D shadow map from Z6 tiles")
    parser.add_argument("dir", help="Tile directory")
    parser.add_argument("--out", default="shadow.ktx2", help="Output filename")
    parser.add_argument("--zoom", default=7, type=int, help="Zoom level")
    parser.add_argument(
        "--bbox", default="9.4,46.2,17.4,49.2", help="lon_min,lat_min,lon_max,lat_max"
    )

    args = parser.parse_args()

    bbox = [float(x) for x in args.bbox.split(",")]
    region = {
        "lon_min": bbox[0],
        "lat_min": bbox[1],
        "lon_max": bbox[2],
        "lat_max": bbox[3],
    }
    out_path = os.path.join(args.dir, args.out)

    config = ShadowConfig(zoom_level=args.zoom)
    generator = ShadowMapGenerator(args.dir, config)
    generator.generate(region, out_path)


if __name__ == "__main__":
    main()
