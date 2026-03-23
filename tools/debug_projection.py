import os
import argparse
import re
import sys
import numpy as np

try:
    from PIL import Image
except ImportError:
    print("Error: PIL is missing. Run: pip install Pillow==12.1.1")
    sys.exit(1)


# Import your existing utilities
from io_ktx import Ktx2
from bc4 import BC4Compressor


class CloudMapGenerator:
    TILE_RES = 256

    def __init__(self, input_path, zoom, min_h, max_h, layers, out_prefix, max_res, print_stats, norm_max=None):
        self.input_path = input_path
        self.zoom = zoom
        self.min_h = min_h
        self.max_h = max_h
        self.layers = layers
        self.out_prefix = out_prefix
        self.max_res = max_res
        self.print_stats = print_stats
        self.norm_max = norm_max

    def run(self):
        if os.path.isfile(self.input_path):
            self.process_single_file()
        elif os.path.isdir(self.input_path):
            self.process_directory()
        else:
            print(f"Error: Input {self.input_path} not found.")

    def process_single_file(self):
        print(f"Processing single file: {self.input_path}")
        img_layers, stats = self._process_tile_layers(self.input_path)

        if not img_layers:
            return

        base_name = os.path.basename(self.input_path).replace(".ktx2", "")

        # Print aggregated stats for this single file (voxel-level stats)
        if stats is not None:
            print(f"\nAggregated Tile Stats for {base_name}:")
            print(f"  Voxels: {stats['voxel_count']}")
            print(f"  Min (phys): {stats['min']:.6f}")
            print(f"  Max (phys): {stats['max']:.6f}")
            print(f"  Mean (phys): {(stats['sum']/stats['voxel_count']):.6f}")
            print(f"  Variance (phys): {(stats['sumsq']/stats['voxel_count'] - (stats['sum']/stats['voxel_count'])**2):.6f}")
            print(f"  Non-zero Count: {stats['nonzero_count']}")
            near = stats.get('near_max_norm_count', 0)
            pct = (near / stats['voxel_count']) * 100 if stats['voxel_count']>0 else 0.0
            print(f"  Values >95% of representable (vol_norm): {near} ({pct:.2f}%)")

        for i, img_data in enumerate(img_layers):
            self._save_image(img_data, f"{self.out_prefix}_{base_name}_L{i}.png")

    def process_directory(self):
        print(f"Scanning {self.input_path} for Zoom {self.zoom}...")

        # 1. Scan files
        tiles = []  # (x, y, filepath)
        files = os.listdir(self.input_path)
        pattern = re.compile(f"tile_{self.zoom}_(\d+)_(\d+).ktx2$")

        for f in files:
            m = pattern.match(f)
            if m:
                tiles.append(
                    (int(m.group(1)), int(m.group(2)), os.path.join(self.input_path, f))
                )

        if not tiles:
            print(f"No density tiles found for zoom {self.zoom}.")
            return

        # 2. Calculate Canvas Bounds
        xs = [t[0] for t in tiles]
        ys = [t[1] for t in tiles]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        width_tiles = (max_x - min_x) + 1
        height_tiles = (max_y - min_y) + 1

        px_w = width_tiles * self.TILE_RES
        px_h = height_tiles * self.TILE_RES

        print(f"Grid: {min_x},{min_y} to {max_x},{max_y}")
        print(f"Canvas Size: {px_w}x{px_h} pixels")

        # 3. Process Layers
        canvases = [
            np.zeros((px_h, px_w), dtype=np.float32) for _ in range(self.layers)
        ]

        total_tiles = len(tiles)
        print(f"Stitching {total_tiles} tiles...")

        # Initialize aggregation for voxel-level stats across tiles
        agg_voxels = 0
        agg_sum = 0.0
        agg_sumsq = 0.0
        agg_min = None
        agg_max = None
        agg_nonzero = 0
        agg_near_max_norm = 0

        for i, (tx, ty, path) in enumerate(tiles):
            # Progress Report
            if i % 10 == 0 or i == total_tiles - 1:
                percent = ((i + 1) / total_tiles) * 100
                sys.stdout.write(
                    f"\r  Progress: {i + 1}/{total_tiles} ({percent:.1f}%)"
                )
                sys.stdout.flush()

            tile_layers, tile_stats = self._process_tile_layers(path)

            if tile_layers is None:
                continue

            # Aggregate voxel-level stats
            if tile_stats is not None:
                vc = tile_stats['voxel_count']
                agg_voxels += vc
                agg_sum += tile_stats['sum']
                agg_sumsq += tile_stats['sumsq']
                agg_nonzero += tile_stats['nonzero_count']
                agg_near_max_norm += tile_stats.get('near_max_norm_count', 0)
                if agg_min is None or tile_stats['min'] < agg_min:
                    agg_min = tile_stats['min']
                if agg_max is None or tile_stats['max'] > agg_max:
                    agg_max = tile_stats['max']

            # Calculate Canvas Position
            # X: Standard Left-to-Right
            col = tx - min_x

            # Y: TMS (South-Up) -> Image (North-Down) conversion
            # Canvas row 0 is the Top. In TMS, the Top row has the highest Y index.
            row = max_y - ty

            px_x = col * self.TILE_RES
            px_y = row * self.TILE_RES

            for l_idx, layer_data in enumerate(tile_layers):
                canvases[l_idx][
                    px_y : px_y + self.TILE_RES, px_x : px_x + self.TILE_RES
                ] = layer_data

        print("\nNormalization and saving...")
        # Print aggregated voxel-level stats across all tiles (based on vol_norm/vol_phys)
        if agg_voxels > 0 and self.print_stats:
            agg_mean = agg_sum / agg_voxels
            agg_var = agg_sumsq / agg_voxels - (agg_mean ** 2)
            near_pct = (agg_near_max_norm / agg_voxels) * 100
            print(f"\nAggregated Voxels: {agg_voxels}")
            print(f"  Min (phys): {agg_min:.6f}")
            print(f"  Max (phys): {agg_max:.6f}")
            print(f"  Mean (phys): {agg_mean:.6f}")
            print(f"  Variance (phys): {agg_var:.6f}")
            print(f"  Non-zero Count: {agg_nonzero}")
            print(f"  Values >= 95% of representable (vol_norm): {agg_near_max_norm} ({near_pct:.2f}%)")
        # Print aggregated stats for the stitched canvases
        # Per-layer stats
        if self.print_stats:
            for i, canvas in enumerate(canvases):
                self._print_stats(canvas, f"Zoom{self.zoom} Layer {i}")

        # Composite across all layers
        try:
            stacked = np.stack(canvases, axis=0)
            composite = np.sum(stacked, axis=0)
            if self.print_stats:
                self._print_stats(composite, f"Zoom{self.zoom} Composite")
        except Exception:
            pass

        for i, canvas in enumerate(canvases):
            self._save_image(canvas, f"{self.out_prefix}_Z{self.zoom}_layer{i}.png")

    def _process_tile_layers(self, filepath):
        try:
            mip = 0
            mips, meta = Ktx2.load(filepath)
            data_bytes = mips[mip]

            # --- 1. Decompress ---
            if meta.get("_ktx_format", "").startswith("BC4"):
                w, h, d = meta.get("_ktx_block_dimensions", (256, 256, 140))
                w >>= mip
                h >>= mip
                d >>= mip
                # Static method call
                vol_raw = BC4Compressor.decompress(data_bytes, w, h, d)
            else:
                return None

            # --- 2. Decode Physics ---
            if vol_raw.dtype == np.uint8:
                vol_norm = vol_raw.astype(np.float32) / 255.0
            else:
                vol_norm = vol_raw.astype(np.float32)

            val_scale = float(meta.get("val_scale", 64.0))

            # Linear Density = (Stored * Scale) ^ 4
            vol_phys = vol_norm * val_scale

            # Compute simple voxel-level stats to return for aggregation
            voxel_count = vol_phys.size
            v_min = float(np.min(vol_phys)) if voxel_count > 0 else 0.0
            v_max = float(np.max(vol_phys)) if voxel_count > 0 else 0.0
            v_sum = float(np.sum(vol_phys)) if voxel_count > 0 else 0.0
            v_sumsq = float(np.sum(np.square(vol_phys))) if voxel_count > 0 else 0.0
            v_nonzero = int(np.count_nonzero(vol_phys > 0))

            # Count voxels where vol_norm >= 0.95 (representable max basis)
            near_max_norm = int(np.count_nonzero(vol_norm >= 0.95))

            tile_stats = {
                'voxel_count': voxel_count,
                'min': v_min,
                'max': v_max,
                'sum': v_sum,
                'sumsq': v_sumsq,
                'nonzero_count': v_nonzero,
                'near_max_norm_count': near_max_norm,
            }

            # --- 3. Slicing & Integration ---
            phys_res_z = float(meta.get("phys_res_z", 100.0))
            nz, ny, nx = vol_phys.shape

            total_range = self.max_h - self.min_h
            layer_step = total_range / self.layers

            results = []

            for l in range(self.layers):
                h_start = self.min_h + (l * layer_step)
                h_end = h_start + layer_step

                z_start = max(0, int(h_start / phys_res_z))
                z_end = min(nz, int(h_end / phys_res_z))

                if z_start >= z_end:
                    results.append(np.zeros((ny, nx), dtype=np.float32))
                    continue

                # Sum along Z axis
                projection = np.sum(vol_phys[z_start:z_end, :, :], axis=0)

                # Flip vertically because TMS tile content is stored upside-down relative to screen space
                projection = np.flipud(projection)

                results.append(projection)

            return results, tile_stats

        except Exception as e:
            print(f"\nError processing {filepath}: {e}")
            return None, None

    def _print_stats(self, data, label="Data"):
        try:
            arr = np.array(data, dtype=np.float64)
            print(f"\nStatistics for: {label}")
            print(f"  Min: {np.min(arr):.6f}")
            print(f"  Max: {np.max(arr):.6f}")
            print(f"  Mean: {np.mean(arr):.6f}")
            print(f"  Variance: {np.var(arr):.6f}")

            nonzero = arr[arr > 0]
            if nonzero.size > 0:
                print(f"  Non-zero Median: {np.median(nonzero):.6f}")
                print(f"  Non-zero Count: {nonzero.size}")
            else:
                print(f"  Non-zero Median: N/A")
        except Exception as e:
            print(f"  Could not compute stats for {label}: {e}")

    def _save_image(self, data, filename):
        max_val = np.max(data)
        if max_val <= 0:
            print(f"Layer empty, skipping {filename}")
            return

        # Visualization Curve (Sqrt)
        visual = np.power(data, 0.5)

        # Normalize to 0-255
        v_max = self.norm_max if self.norm_max is not None else np.max(visual)
        if v_max > 0:
            visual = np.clip(visual / v_max, 0.0, 1.0) * 255.0

        img = Image.fromarray(visual.astype(np.uint8), mode="L")

        # Downsample if needed
        if self.max_res is not None:
            w, h = img.size
            if w > self.max_res or h > self.max_res:
                img.thumbnail((self.max_res, self.max_res), Image.Resampling.LANCZOS)
                print(f"  Downsampled to {img.size}")

        img.save(filename)
        print(f"Saved {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Layered Cloud Map Generator")
    parser.add_argument(
        "input_path", help="Directory containing tiles OR path to single file"
    )
    parser.add_argument("--zoom", type=int, default=11, help="Zoom level for dir mode")
    parser.add_argument("--min-height", type=float, default=0.0)
    parser.add_argument("--max-height", type=float, default=22500.0)
    parser.add_argument(
        "--layers", type=int, default=1, help="Number of vertical layers"
    )
    parser.add_argument("--out-prefix", default="cloud_map")
    parser.add_argument(
        "--max-res",
        type=int,
        default=32767,
        help="Max dimension of output image (pixels)",
    )
    parser.add_argument("--stats", action="store_true", help="Print detailed statistics about the data")
    parser.add_argument(
        "--norm-max",
        type=float,
        default=None,
        help="Max value for visualization normalization (maps to 255). If not specified, uses max of data",
    )

    args = parser.parse_args()

    gen = CloudMapGenerator(
        args.input_path,
        args.zoom,
        args.min_height,
        args.max_height,
        args.layers,
        args.out_prefix,
        args.max_res,
        args.stats,
        args.norm_max,
    )
    gen.run()
