#############################################################################
# weBIGeo Clouds
# Copyright (C) 2026 Wendelin Muth
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
import os
import json
import zstandard as zstd
import numpy as np
import threading
import multiprocessing
import argparse
import traceback
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from bc4 import BC4Compressor
from io_ktx import Ktx2
from numba import jit
from util import report_progress

logger = logging.getLogger("lod")


@dataclass
class LODConfig:
    """Configuration for LOD generation matching tile processor output."""

    tile_resolution: int = 256
    vertical_layers: int = 64
    # Used for normalization, values above will be clipped.
    # 32.0 was chosen because above that visibility is so short range that it doesn't make much of a difference. Could go even lower still.
    max_density_value: float = 32.0
    # Stop generating mipmaps when largest dimension reaches this
    min_mipmap_size: int = 4
    # Whether to analyze compression/quantization error statistics
    analyze_error: bool = False

    @property
    def tile_shape(self) -> tuple:
        """Shape of a single tile (Z, Y, X)."""
        return (self.vertical_layers, self.tile_resolution, self.tile_resolution)

    @property
    def tile_volume_size(self) -> int:
        """Total voxels in a tile."""
        return self.vertical_layers * self.tile_resolution * self.tile_resolution


class RawTileLoader:
    """Loads and parses compressed raw tile files."""

    @staticmethod
    def load_and_optionally_delete(
        filepath: str, config: LODConfig, delete: bool = True
    ) -> dict:
        """Load a raw .zst tile file and optionally delete it.

        Returns:
            dict with keys: 'density' (float32 array), 'meta' (dict)
            or None if file doesn't exist or fails to load
        """
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, "rb") as f:
                decompressor = zstd.ZstdDecompressor()
                with decompressor.stream_reader(f) as reader:
                    data = reader.read()

            # Parse header
            header_length = int.from_bytes(data[:4], "little")
            metadata = json.loads(data[4 : 4 + header_length])
            data_offset = 4 + header_length

            # Extract density bytes
            density_bytes = data[
                data_offset : data_offset + config.tile_volume_size * 2
            ]
            density_f16 = np.frombuffer(density_bytes, dtype=np.float16).reshape(
                config.tile_shape
            )
            density_f32 = density_f16.astype(np.float32)

            if delete:
                os.remove(filepath)

            return {"density": density_f32, "meta": metadata}

        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}", exc_info=True)
            return None


class VolumeDownsampler:
    """Handles 3D volume downsampling for mipmaps and LOD generation."""

    @staticmethod
    def downsample_single_volume(volume: np.ndarray) -> np.ndarray:
        """Downsample a 3D volume by 2x using mean pooling.

        Args:
            volume: Input volume with shape (D, H, W)

        Returns:
            Downsampled volume with shape (D//2, H//2, W//2)
        """
        d, h, w = volume.shape
        new_d, new_h, new_w = d // 2, h // 2, w // 2

        # Crop to even dimensions
        cropped = volume[: new_d * 2, : new_h * 2, : new_w * 2]

        # Reshape to separate even/odd indices
        reshaped = cropped.reshape(new_d, 2, new_h, 2, new_w, 2)

        # Average over the '2' dimensions
        return reshaped.mean(axis=(1, 3, 5))

    @staticmethod
    def downsample_tile_grid(
        child_tiles: list, tile_shape: tuple, mode: str = "mean"
    ) -> np.ndarray:
        """Downsample 4 child tiles into 1 parent tile.

        Args:
            child_tiles: List of 4 tiles [TL, TR, BL, BR], each can be None or ndarray
            tile_shape: Shape of each tile (nz, ny, nx)
            mode: Downsampling mode ('mean' or 'min')

        Returns:
            Downsampled parent tile or None if all children are None
        """
        nz, ny, nx = tile_shape
        canvas = np.zeros((nz, ny * 2, nx * 2), dtype=np.float32)

        # Tile positions: TL, TR, BL, BR
        positions = [(0, 0), (0, nx), (ny, 0), (ny, nx)]

        has_data = False
        for tile_data, (offset_y, offset_x) in zip(child_tiles, positions):
            if tile_data is not None:
                has_data = True
                canvas[:, offset_y : offset_y + ny, offset_x : offset_x + nx] = (
                    tile_data
                )

        if not has_data:
            return None

        # Reshape for 2x2 pooling
        reshaped = canvas.reshape(nz, ny, 2, nx, 2)

        if mode == "min":
            return reshaped.min(axis=(2, 4))
        else:
            return reshaped.mean(axis=(2, 4))


class MipmapGenerator:
    """Generates mipmaps with BC4 compression for KTX2 output."""

    def __init__(self, config: LODConfig):
        self.config = config
        self.thread_local = threading.local()

    @property
    def compressor(self) -> BC4Compressor:
        """Thread-local BC4 compressor instance."""
        if not hasattr(self.thread_local, "bc4_compressor"):
            self.thread_local.bc4_compressor = BC4Compressor()
        return self.thread_local.bc4_compressor

    def generate_mipmaps(self, density: np.ndarray) -> list:
        """Generate mipmap chain with BC4 compression.

        Args:
            density: Base level density data (float32)

        Returns:
            List of compressed mipmap levels (bytes)
        """
        compressed_levels = []
        current_volume = density.astype(np.float32) / self.config.max_density_value

        while True:
            normalized = np.clip(current_volume, 0.0, 1.0)

            # Compress with BC4
            compressed = self.compressor.compress(normalized, is_volume=True)
            compressed_levels.append(compressed)

            # Check if we should stop
            max_dimension = max(current_volume.shape)
            if max_dimension <= self.config.min_mipmap_size:
                break

            # Downsample for next level
            current_volume = VolumeDownsampler.downsample_single_volume(current_volume)

            # Safety check
            if any(dim < 1 for dim in current_volume.shape):
                break

        return compressed_levels


class TileCoordinateSystem:
    """Handles tile coordinate transformations (XYZ to TMS)."""

    @staticmethod
    def xyz_to_tms(x: int, y: int, z: int) -> tuple:
        """Convert XYZ tile coordinates to TMS (flip Y axis).

        Args:
            x, y, z: XYZ tile coordinates

        Returns:
            (x, y_tms, z) tuple
        """
        y_tms = (1 << z) - 1 - y
        return (x, y_tms, z)

    @staticmethod
    def build_filepath(directory: str, x: int, y: int, z: int, extension: str) -> str:
        """Build filepath for a tile."""
        return os.path.join(directory, f"tile_{z}_{x}_{y}{extension}")


class ProgressTracker:
    """Thread-safe progress tracking."""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.lock = threading.Lock()

    def increment(self):
        """Increment progress counter and print status."""
        with self.lock:
            self.completed += 1
            percent = (self.completed / self.total) * 100 if self.total > 0 else 0
            report_progress(
                "lod_generation",
                f"{self.completed}/{self.total} tiles generated",
                percent,
            )

    def finish(self):
        pass


class LODTreeBuilder:
    """Builds the complete LOD tree structure from leaf tiles."""

    @staticmethod
    def calculate_all_tiles(leaf_coords: set, start_zoom: int, max_zoom: int) -> set:
        """Calculate all tiles needed in the LOD tree.

        Args:
            leaf_coords: Set of (x, y) coordinates at max_zoom
            start_zoom: Top level of the tree
            max_zoom: Bottom level (leaves)

        Returns:
            Set of all (x, y) coordinates across all zoom levels
        """
        all_tiles = set(leaf_coords)
        current_level = set(leaf_coords)

        for zoom in range(max_zoom - 1, start_zoom - 1, -1):
            parent_level = set()
            for x, y in current_level:
                parent_level.add((x // 2, y // 2))
            all_tiles.update(parent_level)
            current_level = parent_level

        return all_tiles

    @staticmethod
    def find_root_tiles(leaf_coords: set, start_zoom: int, max_zoom: int) -> set:
        """Find root tiles that need to be processed.

        Args:
            leaf_coords: Set of (x, y) coordinates at max_zoom
            start_zoom: Top level of the tree
            max_zoom: Bottom level (leaves)

        Returns:
            Set of (x, y) root coordinates at start_zoom
        """
        zoom_difference = max_zoom - start_zoom
        scale_factor = 1 << zoom_difference

        roots = set()
        for leaf_x, leaf_y in leaf_coords:
            root_x = leaf_x // scale_factor
            root_y = leaf_y // scale_factor
            roots.add((root_x, root_y))

        return roots


class LODGenerator:
    """Main LOD generator with recursive tree traversal."""

    def __init__(
        self,
        data_dir: str,
        start_zoom: int = 4,
        max_zoom: int = 11,
        keep_raw: bool = False,
        config: LODConfig = None,
        max_workers: int = None,
    ):
        self.data_dir = data_dir
        self.start_zoom = start_zoom
        self.max_zoom = max_zoom
        self.keep_raw = keep_raw
        self.config = config or LODConfig()

        max_workers = max_workers or multiprocessing.cpu_count() * 2
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.max_workers = max_workers

        self.active_threads = 0
        self.thread_lock = threading.Lock()
        self.stop_event = threading.Event()

        self.progress_tracker = None
        self.mipmap_generator = MipmapGenerator(self.config)
        # Collect per-tile error statistics for compression/quantization analysis
        self._tile_error_stats = []
        self._tile_p99 = []
        # Trimming percentile for outlier filtering (e.g., 99.9)
        self._trim_percentile = 99.9

    def run(self):
        """Main entry point for LOD generation."""
        logger.info(f"Starting LOD generation (zoom {self.start_zoom}–{self.max_zoom})")
        report_progress("lod_generation", "", 0)

        # Scan for leaf tiles
        leaf_coords = self._scan_leaf_tiles()
        if not leaf_coords:
            logger.warning("No raw leaf files found")
            return

        # Calculate work
        all_tiles = LODTreeBuilder.calculate_all_tiles(
            leaf_coords, self.start_zoom, self.max_zoom
        )
        root_tiles = LODTreeBuilder.find_root_tiles(
            leaf_coords, self.start_zoom, self.max_zoom
        )

        logger.info(f"Found {len(leaf_coords)} leaves, {len(all_tiles)} tiles to generate")

        self.progress_tracker = ProgressTracker(len(all_tiles))

        # Process tree
        try:
            self._process_tree_roots(sorted(root_tiles))
        except KeyboardInterrupt:
            logger.warning("Ctrl+C detected — shutting down")
            self.stop_event.set()
            self.executor.shutdown(wait=False, cancel_futures=True)
            return
        finally:
            self.executor.shutdown(wait=True)

        self.progress_tracker.finish()

        if self.config.analyze_error:
            self._print_error_summary()

    def _scan_leaf_tiles(self) -> set:
        """Scan directory for leaf tile files at max_zoom."""
        logger.debug(f"Scanning {self.data_dir} for zoom level {self.max_zoom} files")

        files = [f for f in os.listdir(self.data_dir) if f.endswith(".raw.zst")]
        leaf_coords = set()

        for filename in files:
            parts = filename.split("_")
            if len(parts) == 4 and parts[1] == str(self.max_zoom):
                x = int(parts[2])
                y = int(parts[3].split(".")[0])
                leaf_coords.add((x, y))

        return leaf_coords

    def _process_tree_roots(self, root_coords: list):
        """Process all root tiles in parallel."""
        with self.thread_lock:
            self.active_threads += len(root_coords)

        futures = [
            self.executor.submit(self._worker_wrapper, self.start_zoom, x, y)
            for x, y in root_coords
        ]

        # Wait with periodic checks for keyboard interrupt
        while True:
            done, pending = wait(futures, timeout=0.5, return_when=ALL_COMPLETED)

            # Check for exceptions
            for future in done:
                if future.exception():
                    raise future.exception()

            if not pending:
                break

    def _worker_wrapper(self, z: int, x: int, y: int):
        """Wrapper to manage thread lifecycle and exceptions."""
        try:
            if self.stop_event.is_set():
                return None
            return self._recursive_build(z, x, y)
        except Exception:
            logger.error(f"Error at {z}/{x}/{y}", exc_info=True)
            return None
        finally:
            with self.thread_lock:
                self.active_threads -= 1

    def _recursive_build(self, z: int, x: int, y: int) -> dict:
        """Recursively build tile tree with depth-first traversal.

        Returns:
            dict with 'density' and 'meta' keys, or None if tile has no data
        """
        if self.stop_event.is_set():
            return None

        # Base case: Load leaf tile
        if z == self.max_zoom:
            filepath = TileCoordinateSystem.build_filepath(
                self.data_dir, x, y, z, ".raw.zst"
            )
            tile_data = RawTileLoader.load_and_optionally_delete(
                filepath, self.config, delete=not self.keep_raw
            )

            if tile_data:
                self._save_ktx_tile(z, x, y, tile_data["density"], tile_data["meta"])

            return tile_data

        # Recursive case: Process children
        child_z = z + 1
        child_base_x = x * 2
        child_base_y = y * 2

        # Process 4 children: TL, TR, BL, BR
        child_coords = [
            (child_base_x, child_base_y),
            (child_base_x + 1, child_base_y),
            (child_base_x, child_base_y + 1),
            (child_base_x + 1, child_base_y + 1),
        ]

        children = [None] * 4
        pending_futures = {}

        # Dispatch children (fork to thread pool if capacity available)
        for i, (cx, cy) in enumerate(child_coords):
            should_fork = False

            with self.thread_lock:
                if self.active_threads < self.max_workers:
                    self.active_threads += 1
                    should_fork = True

            if should_fork:
                if self.stop_event.is_set():
                    return None
                future = self.executor.submit(self._worker_wrapper, child_z, cx, cy)
                pending_futures[i] = future
            else:
                # Run inline (depth-first)
                children[i] = self._recursive_build(child_z, cx, cy)

        # Collect forked results
        for i, future in pending_futures.items():
            children[i] = future.result()

        # If all children are empty, this tile is empty
        if all(child is None for child in children):
            return None

        # Extract density data
        child_densities = [child["density"] if child else None for child in children]

        # Get metadata from first valid child
        first_valid = next(child for child in children if child is not None)
        parent_meta = first_valid["meta"].copy()
        parent_meta["phys_res_x"] *= 2.0
        parent_meta["phys_res_y"] *= 2.0

        # Downsample to create parent tile
        parent_density = VolumeDownsampler.downsample_tile_grid(
            child_densities, self.config.tile_shape, mode="mean"
        )

        # Save parent tile
        self._save_ktx_tile(z, x, y, parent_density, parent_meta)

        return {"density": parent_density, "meta": parent_meta}

    def _save_ktx_tile(
        self, z: int, x: int, y: int, density: np.ndarray, metadata: dict
    ):
        """Save tile as KTX2 with mipmaps and BC4 compression."""
        if density is None:
            return

        # Convert to TMS coordinates
        x_tms, y_tms, z_tms = TileCoordinateSystem.xyz_to_tms(x, y, z)
        filepath = TileCoordinateSystem.build_filepath(
            self.data_dir, x_tms, y_tms, z_tms, ".ktx2"
        )

        # Flip Y axis for texture coordinate system
        density = np.flip(density, axis=1)

        # Generate mipmaps
        compressed_mipmaps = self.mipmap_generator.generate_mipmaps(density)

        # Prepare metadata
        ktx_metadata = metadata.copy()
        ktx_metadata.update(
            {
                "val_scale": self.config.max_density_value,
                "tile_z": z_tms,
                "tile_x": x_tms,
                "tile_y": y_tms,
                "compression": "BC4",
            }
        )

        # Save KTX2
        Ktx2.save(
            data=compressed_mipmaps,
            path=filepath,
            target_format="BC4_UNORM_BLOCK",
            metadata=ktx_metadata,
            shape=self.config.tile_shape,
            compress=True,
        )

        # Analyze compression/quantization error for the base level
        if self.config.analyze_error:
            try:
                nz, ny, nx = density.shape

                # Decompress base level (normalized) and rescale
                decompressed_norm = BC4Compressor.decompress(
                    compressed_mipmaps[0], nx, ny, nz
                )

                reconstructed = decompressed_norm.astype(np.float32) * self.config.max_density_value

                # Metrics
                total_voxels = float(density.size)
                clipped_count = int(np.count_nonzero(density > self.config.max_density_value))
                clipped_fraction = clipped_count / total_voxels if total_voxels > 0 else 0.0

                abs_err = np.abs(reconstructed - density)
                max_abs = float(np.max(abs_err))
                mean_abs = float(np.mean(abs_err))
                mse = float(np.mean(abs_err ** 2))
                zero_loss = int(np.count_nonzero((reconstructed == 0.0) & (density > 0.0)))
                zero_loss_fraction = zero_loss / total_voxels if total_voxels > 0 else 0.0

                # Per-tile high-percentile to help recommend a sensible max_density_value
                try:
                    p99 = float(np.percentile(density, 99.9))
                except Exception:
                    p99 = float(np.max(density))

                # Robust per-tile error summaries
                median_abs = float(np.median(abs_err))
                p99_abs = float(np.percentile(abs_err, 99.0))
                p999_abs = float(np.percentile(abs_err, 99.9))
                rmse = float(np.sqrt(mse))

                self._tile_error_stats.append(
                    {
                        "tile": (z, x, y),
                        "max_abs": max_abs,
                        "median_abs": median_abs,
                        "p99_abs": p99_abs,
                        "p999_abs": p999_abs,
                        "mean_abs": mean_abs,
                        "rmse": rmse,
                        "mse": mse,
                        "clipped_fraction": clipped_fraction,
                        "zero_loss_fraction": zero_loss_fraction,
                        "p99": p99,
                    }
                )
                self._tile_p99.append(p99)
            except Exception as e:
                logger.warning(f"Failed to compute error metrics for {filepath}: {e}")

        self.progress_tracker.increment()

    def _print_error_summary(self):
        """Aggregate and log compression/quantization error statistics."""
        if not self._tile_error_stats:
            logger.info("No compression error statistics collected")
            return

        # Collect arrays of per-tile robust metrics
        max_abs_vals = np.array([t["max_abs"] for t in self._tile_error_stats])
        median_abs_vals = np.array([t["median_abs"] for t in self._tile_error_stats])
        p99_abs_vals = np.array([t["p99_abs"] for t in self._tile_error_stats])
        p999_abs_vals = np.array([t["p999_abs"] for t in self._tile_error_stats])
        mean_abs_vals = np.array([t["mean_abs"] for t in self._tile_error_stats])
        rmse_vals = np.array([t["rmse"] for t in self._tile_error_stats])
        mse_vals = np.array([t["mse"] for t in self._tile_error_stats])
        clipped_fracs = np.array([t["clipped_fraction"] for t in self._tile_error_stats])
        zero_loss_fracs = np.array([t["zero_loss_fraction"] for t in self._tile_error_stats])

        overall_max_abs = float(np.max(max_abs_vals))
        overall_median_abs = float(np.median(median_abs_vals))
        overall_p99_abs = float(np.median(p99_abs_vals))
        overall_trimmed_max_abs = float(np.percentile(max_abs_vals, self._trim_percentile))
        overall_rmse = float(np.mean(rmse_vals))
        overall_mse = float(np.mean(mse_vals))
        overall_clipped_frac = float(np.mean(clipped_fracs))
        overall_zero_loss_frac = float(np.mean(zero_loss_fracs))

        # Recommend a max_density_value based on median of per-tile 99.9th percentiles
        try:
            recommended = float(np.median(self._tile_p99))
        except Exception:
            recommended = float(np.max(self._tile_p99))

        # Fraction of tiles whose p99 exceeds current config limit
        tiles_exceeding = float(np.count_nonzero(np.array(self._tile_p99) > self.config.max_density_value))
        tiles_fraction_exceeding = tiles_exceeding / len(self._tile_p99)

        top_n = 5
        sorted_by_p99 = sorted(self._tile_error_stats, key=lambda t: t["p99"], reverse=True)[:top_n]
        sorted_by_clipped = sorted(self._tile_error_stats, key=lambda t: t["clipped_fraction"], reverse=True)[:top_n]

        top_p99_lines = "\n".join(
            f"  {t['tile']}: p99={t['p99']:.3f}, p99_abs={t['p99_abs']:.6f}, clipped={t['clipped_fraction']:.3%}"
            for t in sorted_by_p99
        )
        top_clipped_lines = "\n".join(
            f"  {t['tile']}: clipped={t['clipped_fraction']:.3%}, p99={t['p99']:.3f}"
            for t in sorted_by_clipped
        )

        logger.info(
            f"Compression/Quantization Summary\n"
            f"  Tiles analyzed:              {len(self._tile_error_stats)}\n"
            f"  Current max_density_value:   {self.config.max_density_value}\n"
            f"  Recommended max_density_value (median tile 99.9pct): {recommended:.3f}\n"
            f"  Tiles above current limit:   {tiles_fraction_exceeding:.3%}\n"
            f"  Overall median abs error:    {overall_median_abs:.6f}\n"
            f"  Overall p99 abs error:       {overall_p99_abs:.6f}\n"
            f"  Overall trimmed max abs:     {overall_trimmed_max_abs:.6f}\n"
            f"  Overall RMSE:                {overall_rmse:.6f}\n"
            f"  Overall MSE:                 {overall_mse:.6f}\n"
            f"  Overall clipped fraction:    {overall_clipped_frac:.3%}\n"
            f"  Overall zero-loss fraction:  {overall_zero_loss_frac:.3%}\n"
            f"Top tiles by p99:\n{top_p99_lines}\n"
            f"Top tiles by clipped fraction:\n{top_clipped_lines}"
        )



def main():
    parser = argparse.ArgumentParser(
        description="Generate LOD pyramid for cloud tiles with mipmaps."
    )
    parser.add_argument("path", help="Path to tile directory")
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep intermediate raw files (don't delete after processing)",
    )
    parser.add_argument(
        "--tile-res", type=int, default=256, help="Tile resolution (default: 256)"
    )
    parser.add_argument(
        "--vertical-layers",
        type=int,
        default=64,
        help="Number of vertical layers",
    )
    parser.add_argument(
        "--start-zoom", type=int, default=6, help="Top zoom level (default: 6)"
    )
    parser.add_argument(
        "--max-zoom",
        type=int,
        default=11,
        help="Bottom zoom level with source data (default: 11)",
    )

    args = parser.parse_args()

    config = LODConfig(
        tile_resolution=args.tile_res, vertical_layers=args.vertical_layers
    )

    generator = LODGenerator(
        data_dir=args.path,
        start_zoom=args.start_zoom,
        max_zoom=args.max_zoom,
        keep_raw=args.keep_raw,
        config=config,
    )

    generator.run()


if __name__ == "__main__":
    main()
