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
import math
import time
import json
import zstandard as zstd
import numpy as np
import wgpu
import mercantile
import traceback
import gc
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from .icon_loader import DataCube
import config as _server_config
from utils.general import report_progress
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger("tiles")


# --- CONFIGURATION ---
@dataclass
class TileConfig:
    """Configuration for tile processing and output resolution."""

    tile_resolution: int = 256
    halo_fraction: float = 0.125  # Halo as fraction of tile_resolution (32/256 = 0.125)
    vertical_layers: int = 64
    batch_tiles_x: int = 4
    batch_tiles_y: int = 4

    @property
    def halo_size(self) -> int:
        """Calculate halo size as a fraction of tile resolution."""
        return int(self.tile_resolution * self.halo_fraction)

    @property
    def batch_width(self) -> int:
        """Full width including halos."""
        return self.batch_tiles_x * self.tile_resolution + 2 * self.halo_size

    @property
    def batch_height(self) -> int:
        """Full height including halos."""
        return self.batch_tiles_y * self.tile_resolution + 2 * self.halo_size

    @property
    def batch_buffer_size(self) -> int:
        """Size in bytes for a full batch buffer."""
        return self.batch_width * self.batch_height * self.vertical_layers * 4


class GPUPipelineSlot:
    """
    Manages GPU resources for a single execution lane.
    Implements double buffering: one slot computes while the other reads back.
    """

    def __init__(
        self,
        device: wgpu.GPUDevice,
        main_bind_group_layouts: list,
        postprocess_bind_group_layouts: list,
        buffer_size: int,
    ):
        self.device = device
        self.is_busy = False
        self.metadata = None

        # Uniform buffer for synthesis parameters (64 bytes)
        self.uniform_buffer = device.create_buffer(
            size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

        # Storage buffer for high-resolution extinction output
        self.main_output_buffer = device.create_buffer(
            size=buffer_size, usage=wgpu.BufferUsage.STORAGE
        )
        self.final_output_buffer = device.create_buffer(
            size=buffer_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )

        # Bind groups for shader access
        self.main_uniform_bind_group = device.create_bind_group(
            layout=main_bind_group_layouts[1],
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.uniform_buffer,
                        "offset": 0,
                        "size": 64,
                    },
                }
            ],
        )

        self.main_output_bind_group = device.create_bind_group(
            layout=main_bind_group_layouts[2],
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.main_output_buffer,
                        "offset": 0,
                        "size": buffer_size,
                    },
                }
            ],
        )

        # Postprocess bind groups (input is main output, output is final output)

        self.final_input_bind_group = device.create_bind_group(
            layout=postprocess_bind_group_layouts[0],
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.main_output_buffer,
                        "offset": 0,
                        "size": buffer_size,
                    },
                }
            ],
        )

        self.final_uniform_bind_group = device.create_bind_group(
            layout=postprocess_bind_group_layouts[1],
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.uniform_buffer,
                        "offset": 0,
                        "size": 64,
                    },
                }
            ],
        )

        self.final_output_bind_group = device.create_bind_group(
            layout=postprocess_bind_group_layouts[2],
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.final_output_buffer,
                        "offset": 0,
                        "size": buffer_size,
                    },
                }
            ],
        )


class CloudDataUploader:
    """Handles uploading meteorological data to GPU textures."""

    # Data fields to upload (order matters for shader binding)
    DATA_FIELDS = ["clc", "qc", "qi", "qv", "t", "p", "tke", "hhl"]

    @staticmethod
    def upload_datacube(device: wgpu.GPUDevice, datacube) -> tuple:
        """Upload all required data fields as 3D textures."""
        logger.info("Uploading source data to GPU (Float16)")

        texture_views = []
        for name in CloudDataUploader.DATA_FIELDS:
            data = datacube.data[name]
            nz, ny, nx = data.shape
            logger.debug(f"  {name}: shape={data.shape}, dtype={data.dtype}")
            texture = device.create_texture(
                size=(nx, ny, nz),
                usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
                dimension="3d",
                format="r16float",
            )
            device.queue.write_texture(
                {"texture": texture},
                np.ascontiguousarray(data, dtype=np.float16),
                {"bytes_per_row": nx * 2, "rows_per_image": ny},
                (nx, ny, nz),
            )
            texture_views.append(texture.create_view())

        # Extract dimensions
        nz_height_levels, ny_src, nx_src = datacube.data["hhl"].shape
        nz_src = nz_height_levels - 1

        return texture_views, (nx_src, ny_src, nz_src, nz_height_levels)


class ShaderPipelineBuilder:
    """Builds GPU compute pipelines for cloud synthesis."""

    @staticmethod
    def create_synthesis_pipeline(device: wgpu.GPUDevice, shader_path: str) -> tuple:
        """Create compute pipeline and bind group layouts."""
        with open(shader_path, "r") as f:
            shader_code = f.read()

        shader_module = device.create_shader_module(code=shader_code)

        # Layout 0: Input textures (data fields)
        texture_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": i,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "texture": {
                        "sample_type": "unfilterable-float",
                        "view_dimension": "3d",
                    },
                }
                for i in range(len(CloudDataUploader.DATA_FIELDS))
            ]
        )

        # Layout 1: Uniform buffer (parameters)
        uniform_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "uniform"},
                }
            ]
        )

        # Layout 2: Storage buffer (output)
        storage_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                }
            ]
        )

        layouts = [texture_layout, uniform_layout, storage_layout]

        pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=layouts),
            compute={"module": shader_module, "entry_point": "main"},
        )

        return pipeline, layouts

    @staticmethod
    def create_postprocess_pipeline(device: wgpu.GPUDevice, shader_path: str) -> tuple:
        """Create compute pipeline and bind group layouts."""
        with open(shader_path, "r") as f:
            shader_code = f.read()

        shader_module = device.create_shader_module(code=shader_code)

        input_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": "storage",
                    },
                }
            ]
        )

        uniform_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "uniform"},
                }
            ]
        )

        output_layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                }
            ]
        )

        layouts = [input_layout, uniform_layout, output_layout]

        pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=layouts),
            compute={"module": shader_module, "entry_point": "main"},
        )

        return pipeline, layouts


class TileBatcher:
    """Creates batches of tiles for efficient GPU processing."""

    @staticmethod
    def create_batches(tiles: list, config: TileConfig) -> list:
        """Group tiles into batches for parallel processing."""
        tile_map = {(t.x, t.y): t for t in tiles}

        min_x = min(t.x for t in tiles)
        max_x = max(t.x for t in tiles)
        min_y = min(t.y for t in tiles)
        max_y = max(t.y for t in tiles)

        batches = []
        for batch_x in range(min_x, max_x + 1, config.batch_tiles_x):
            for batch_y in range(min_y, max_y + 1, config.batch_tiles_y):
                batch_tiles = []
                for ix in range(config.batch_tiles_x):
                    for iy in range(config.batch_tiles_y):
                        key = (batch_x + ix, batch_y + iy)
                        if key in tile_map:
                            batch_tiles.append((ix, iy, tile_map[key]))

                if batch_tiles:
                    batches.append((batch_x, batch_y, batch_tiles))

        return batches


class UniformParameterCalculator:
    """Calculates shader uniform parameters for different projections."""

    def __init__(self, lats: np.ndarray, lons: np.ndarray, source_dims: tuple):
        self.lats = lats
        self.lons = lons
        self.lat0 = lats[0]
        self.lon0 = lons[0]
        self.lat_step = (lats[-1] - self.lat0) / (len(lats) - 1)
        self.lon_step = (lons[-1] - self.lon0) / (len(lons) - 1)
        self.nx_src, self.ny_src, self.nz_src, self.nz_hhl = source_dims

    def calculate_for_batch(
        self, batch_x: int, batch_y: int, zoom: int, config: TileConfig
    ) -> np.ndarray:
        """Calculate uniform parameters for a tile batch."""
        tile_bounds = mercantile.xy_bounds(mercantile.Tile(batch_x, batch_y, zoom))
        resolution_h = (tile_bounds.right - tile_bounds.left) / config.tile_resolution
        resolution_v = _server_config.max_altitude / config.vertical_layers

        # Top-left corner with halo
        left = tile_bounds.left - config.halo_size * resolution_h
        top = tile_bounds.top + config.halo_size * resolution_h

        # Convert to lat/lon
        top_left_lon, top_left_lat = mercantile.lnglat(left, top)
        bottom_right_lon, bottom_right_lat = mercantile.lnglat(
            left + config.batch_width * resolution_h,
            top - config.batch_height * resolution_h,
        )

        # Calculate texture space offsets and scales
        offset_x = (top_left_lon - self.lon0) / self.lon_step
        offset_y = (top_left_lat - self.lat0) / self.lat_step
        inv_scale_x = (
            (bottom_right_lon - top_left_lon) / self.lon_step
        ) / config.batch_width
        inv_scale_y = (
            (bottom_right_lat - top_left_lat) / self.lat_step
        ) / config.batch_height

        return np.array(
            [
                self.nx_src,
                self.ny_src,
                self.nz_src,
                self.nz_hhl,
                left,
                top,
                resolution_h,
                resolution_v,
                offset_x,
                offset_y,
                inv_scale_x,
                inv_scale_y,
                config.batch_width,
                config.batch_height,
                config.vertical_layers,
                0.0,
            ],
            dtype=np.float32,
        )


class TileSaver:
    """Handles encoding and saving of processed tile data."""

    COMPRESSION_LEVEL = 3

    @staticmethod
    def save_compressed_tile(
        data: np.ndarray,
        tile,
        resolution_h: float,
        resolution_v: float,
        output_dir: str,
    ):
        """Save a single tile with compression and metadata."""
        try:
            metadata = {
                "phys_res_x": float(resolution_h),
                "phys_res_y": float(resolution_h),
                "phys_res_z": float(resolution_v),
                "tile_z": tile.z,
                "tile_x": tile.x,
                "tile_y": tile.y,
            }

            header_json = json.dumps(metadata).encode("utf-8")
            header_length = len(header_json).to_bytes(4, "little")

            filepath = os.path.join(
                output_dir, f"tile_{tile.z}_{tile.x}_{tile.y}.raw.zst"
            )

            with open(filepath, "wb") as f:
                compressor_context = zstd.ZstdCompressor(
                    level=TileSaver.COMPRESSION_LEVEL
                )
                with compressor_context.stream_writer(f) as compressor:
                    compressor.write(header_length)
                    compressor.write(header_json)
                    compressor.write(data.tobytes())

        except Exception:
            logger.error(f"Error saving tile {tile.z}/{tile.x}/{tile.y}", exc_info=True)


class TileProcessor:
    """Main processor for generating cloud tiles from meteorological data."""

    SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders/cloud_synthesis.wgsl")
    # SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders/cloud_upscale-only.wgsl")
    POSTPROCESS_SHADER_PATH = os.path.join(os.path.dirname(__file__), "shaders/postprocess.wgsl")
    WORKGROUP_SIZE = 8

    def __init__(self, datacube: DataCube, output_dir: str, config: TileConfig = None):
        self.output_dir = output_dir
        self.config = config or TileConfig()
        os.makedirs(output_dir, exist_ok=True)

        self.device = wgpu.utils.get_default_device()
        self.save_thread_pool = ThreadPoolExecutor(max_workers=4)
        self.pending_saves = []
        self.max_pending_saves = 4

        # Store coordinate system
        self.lats = datacube.meta["lat_coords"]
        self.lons = datacube.meta["lon_coords"]

        preprocessed_datacube = self._preprocess_cloud_data(datacube)

        # Upload data to GPU
        self.texture_views, source_dims = CloudDataUploader.upload_datacube(
            self.device, preprocessed_datacube
        )

        del datacube
        del preprocessed_datacube
        gc.collect()

        # Initialize uniform calculator
        self.uniform_calculator = UniformParameterCalculator(
            self.lats, self.lons, source_dims
        )

        # Build GPU pipeline
        self._initialize_gpu_pipeline()

        # Create double-buffered slots
        self.slots = [
            GPUPipelineSlot(
                self.device,
                self.main_bind_group_layouts,
                self.postprocess_bind_group_layouts,
                self.config.batch_buffer_size,
            ),
            GPUPipelineSlot(
                self.device,
                self.main_bind_group_layouts,
                self.postprocess_bind_group_layouts,
                self.config.batch_buffer_size,
            ),
        ]

    def _preprocess_cloud_data(self, datacube: DataCube):
        from .dwd_preprocess import (
            clean_and_remap_clc,
            separate_cirrus_layers,
        )

        # Prepare cloud cover data
        cloud_cover = clean_and_remap_clc(
            datacube.data["clc"].astype(np.float32) * 0.01
        )

        # Extract hydrometeor fields
        qc, qi = [datacube.data[k].astype(np.float32) for k in ["qc", "qi"]]

        # TODO: Unused (Warning: Outdated implementation!)
        # separate_cirrus_layers(cloud_cover, height_levels, qc, qi)

        data_copy = datacube.data.copy()
        data_copy["clc"] = cloud_cover
        data_copy["qc"] = qc * 10000.0  # scale to preserve precision if f16
        data_copy["qi"] = qi * 10000.0
        data_copy["qv"] = datacube.data["qv"].astype(np.float32) * 10000.0
        data_copy["p"] = (
            datacube.data["p"].astype(np.float32) / 1000.0
        )  # Convert Pa to kPa for f16 precision

        # Spatial blur to reduce cell-to-cell patchiness from ICON-D2 QC/QI fields.
        # Separable Gaussian per vertical level in the horizontal (x, y) plane only.
        # sigma=1.0 source voxel gives ~4-6km smoothing, preserving synoptic structure.
        for field in CloudDataUploader.DATA_FIELDS:
            if field in ["hhl"]:
                continue
            arr = data_copy[field]  # shape: (nz, ny, nx)
            arr = gaussian_filter1d(arr, sigma=1.0, axis=2)  # x
            arr = gaussian_filter1d(arr, sigma=1.0, axis=1)  # y
            data_copy[field] = arr

        return DataCube(data_copy, datacube.dims, datacube.meta)

    def _initialize_gpu_pipeline(self):
        """Set up GPU compute pipeline and bind groups."""
        self.main_pipeline, self.main_bind_group_layouts = (
            ShaderPipelineBuilder.create_synthesis_pipeline(
                self.device, self.SHADER_PATH
            )
        )

        self.postprocess_pipeline, self.postprocess_bind_group_layouts = (
            ShaderPipelineBuilder.create_postprocess_pipeline(
                self.device, self.POSTPROCESS_SHADER_PATH
            )
        )

        # Create bind group for input textures
        self.texture_bind_group = self.device.create_bind_group(
            layout=self.main_bind_group_layouts[0],
            entries=[
                {"binding": i, "resource": self.texture_views[i]}
                for i in range(len(CloudDataUploader.DATA_FIELDS))
            ],
        )

    def _dispatch_compute(self, slot: GPUPipelineSlot, width: int, height: int):
        """Record and submit compute commands."""
        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()

        compute_pass.set_pipeline(self.main_pipeline)
        compute_pass.set_bind_group(0, self.texture_bind_group, [], 0, 99)
        compute_pass.set_bind_group(1, slot.main_uniform_bind_group, [], 0, 99)
        compute_pass.set_bind_group(2, slot.main_output_bind_group, [], 0, 99)
        compute_pass.dispatch_workgroups(
            math.ceil(width / self.WORKGROUP_SIZE),
            math.ceil(height / self.WORKGROUP_SIZE),
            1,
        )
        compute_pass.end()

        # postprocess pass
        compute_pass = encoder.begin_compute_pass()

        compute_pass.set_pipeline(self.postprocess_pipeline)
        compute_pass.set_bind_group(0, slot.main_output_bind_group, [], 0, 99)
        compute_pass.set_bind_group(1, slot.final_uniform_bind_group, [], 0, 99)
        compute_pass.set_bind_group(2, slot.final_output_bind_group, [], 0, 99)
        compute_pass.dispatch_workgroups(
            math.ceil(width / self.WORKGROUP_SIZE),
            math.ceil(height / self.WORKGROUP_SIZE),
            1,
        )
        compute_pass.end()

        self.device.queue.submit([encoder.finish()])

    def _wait_for_pending_saves(self):
        """Wait for oldest save to complete if too many are pending."""
        while len(self.pending_saves) >= self.max_pending_saves:
            self.pending_saves[0].result()
            self.pending_saves.pop(0)

    def _process_batch_data(self, data: np.ndarray, batch_metadata: dict):
        """Extract tiles from batch and save them."""
        try:
            tiles = batch_metadata["tiles"]
            volume = data.reshape(
                (
                    self.config.vertical_layers,
                    self.config.batch_height,
                    self.config.batch_width,
                )
            )

            # Get resolution from first tile
            tile_bounds = mercantile.xy_bounds(tiles[0][2])
            resolution_h = (
                tile_bounds.right - tile_bounds.left
            ) / self.config.tile_resolution
            resolution_v = _server_config.max_altitude / self.config.vertical_layers

            for local_x, local_y, tile in tiles:
                # Extract tile region
                start_x = self.config.halo_size + local_x * self.config.tile_resolution
                start_y = self.config.halo_size + local_y * self.config.tile_resolution

                tile_data = volume[
                    :,
                    start_y : start_y + self.config.tile_resolution,
                    start_x : start_x + self.config.tile_resolution,
                ]

                # Save compressed
                TileSaver.save_compressed_tile(
                    tile_data.astype(np.float16),
                    tile,
                    resolution_h,
                    resolution_v,
                    self.output_dir,
                )

                del tile_data

        except Exception:
            logger.error("Error processing batch", exc_info=True)

    def run_tiled(self, zoom: int):
        """Generate tiles at specified zoom level."""
        # Determine coverage
        lat_min, lat_max = sorted([self.lats[0], self.lats[-1]])
        lon_min, lon_max = sorted([self.lons[0], self.lons[-1]])

        tiles = list(mercantile.tiles(lon_min, lat_min, lon_max, lat_max, zoom))
        batches = TileBatcher.create_batches(tiles, self.config)

        logger.info(f"Processing {len(batches)} batches ({len(tiles)} tiles) at zoom {zoom}")
        report_progress("upsampling", "", 0)
        start_time = time.time()
        processed_count = 0

        # Double-buffered execution loop
        for batch_index, batch in enumerate(batches):
            self._wait_for_pending_saves()

            current_slot = self.slots[batch_index % 2]
            previous_slot = self.slots[(batch_index + 1) % 2]

            # Dispatch new batch
            batch_x, batch_y, batch_tiles = batch
            uniforms = self.uniform_calculator.calculate_for_batch(
                batch_x, batch_y, zoom, self.config
            )
            self.device.queue.write_buffer(current_slot.uniform_buffer, 0, uniforms)

            self._dispatch_compute(
                current_slot, self.config.batch_width, self.config.batch_height
            )

            current_slot.is_busy = True
            current_slot.metadata = {"tiles": batch_tiles, "zoom": zoom}

            # Read back previous batch
            if previous_slot.is_busy:
                data_bytes = self.device.queue.read_buffer(
                    previous_slot.final_output_buffer, 0, self.config.batch_buffer_size
                )

                data_array = np.frombuffer(data_bytes, dtype=np.float32).copy()
                del data_bytes

                future = self.save_thread_pool.submit(
                    self._process_batch_data, data_array, previous_slot.metadata
                )
                self.pending_saves.append(future)

                previous_slot.is_busy = False
                processed_count += len(previous_slot.metadata["tiles"])

            # Progress update
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            report_progress(
                "upsampling",
                f"Batch {batch_index + 1}/{len(batches)} ({rate:.1f} tiles/s)",
                (batch_index + 1) / len(batches) * 100,
            )

        # Process final batch
        final_slot = self.slots[(len(batches) - 1) % 2]
        if final_slot.is_busy:
            data = np.frombuffer(
                self.device.queue.read_buffer(
                    final_slot.final_output_buffer, 0, self.config.batch_buffer_size
                ),
                dtype=np.float32,
            ).copy()
            self.save_thread_pool.submit(
                self._process_batch_data, data, final_slot.metadata
            )

        # Wait for all saves to complete
        for future in self.pending_saves:
            future.result()

        self.save_thread_pool.shutdown(wait=True)
