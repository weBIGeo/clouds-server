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

import wgpu
import numpy as np
import os
from numba import njit


@njit(cache=True)
def _decompress_bc4_kernel(bc4_data, depth, blocks_h, blocks_w, height, width):
    """Numba-accelerated kernel for BC4 decompression."""
    output = np.zeros((depth, height, width), dtype=np.float32)

    for d in range(depth):
        for bh in range(blocks_h):
            for bw in range(blocks_w):
                # Calculate block offset
                block_offset = ((d * blocks_h * blocks_w) + (bh * blocks_w) + bw) * 8

                # Endpoints
                r0 = bc4_data[block_offset]
                r1 = bc4_data[block_offset + 1]

                # 1. Build Palette
                p = np.zeros(8, dtype=np.float32)
                p[0] = r0 / 255.0
                p[1] = r1 / 255.0

                if r0 > r1:
                    for i in range(6):
                        # 6 interpolated values
                        p[i + 2] = ((6 - i) * p[0] + (i + 1) * p[1]) / 7.0
                else:
                    for i in range(4):
                        # 4 interpolated values
                        p[i + 2] = ((4 - i) * p[0] + (i + 1) * p[1]) / 5.0
                    p[6] = 0.0
                    p[7] = 1.0

                # 2. Extract 48-bit indices
                # We read 6 bytes manually to avoid endianness/int-size issues in Numba
                idx_bytes = bc4_data[block_offset + 2 : block_offset + 8]

                # Reconstruct the 48-bit integer from little-endian bytes
                indices_int = 0
                for i in range(6):
                    indices_int |= np.uint64(idx_bytes[i]) << (i * 8)

                # 3. Map to 4x4 grid
                for i in range(16):
                    idx = (indices_int >> (i * 3)) & np.uint64(0x07)

                    py = i // 4
                    px = i % 4

                    output[d, bh * 4 + py, bw * 4 + px] = p[idx]

    return output


class BC4Compressor:
    def __init__(self):
        self.device = wgpu.utils.get_default_device()
        self._init_pipeline()

    def _init_pipeline(self):
        shader_path = os.path.join(os.path.dirname(__file__), "shaders/bc4.wgsl")
        with open(shader_path, "r") as f:
            shader_code = f.read()

        self.shader_module = self.device.create_shader_module(code=shader_code)

        self.bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "texture": {
                        "sample_type": "unfilterable-float",
                        "view_dimension": "3d",  # Key change: 3D view
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
            ]
        )

        self.pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout]
        )

        self.compute_pipeline = self.device.create_compute_pipeline(
            layout=self.pipeline_layout,
            compute={"module": self.shader_module, "entry_point": "main"},
        )

    def compress(self, image_np: np.ndarray, is_volume=False) -> bytes:
        """
        Compresses input to BC4.

        Args:
            image_np: Numpy array.
                - If is_volume=False: Expects (H, W).
                - If is_volume=True:  Expects (D, H, W).
            is_volume: Flag to interpret 3D arrays as volumes (D,H,W)
                       instead of raising an error.
        """
        # 1. Handle Input Shapes strictly
        # -------------------------------
        data = image_np

        # Squeeze 'channel' dimension if present (e.g. from cv2.imread usually (H,W,1))
        # We only do this if it's explicitly 1 channel at the end,
        # AND we aren't in volume mode (where last dim is Width).
        if not is_volume and data.ndim == 3 and data.shape[2] == 1:
            data = data.squeeze(2)  # (H, W, 1) -> (H, W)

        if not is_volume:
            if data.ndim != 2:
                raise ValueError(
                    f"For 2D compression, input must be (H, W). Got {data.shape}. "
                    "If you have (H, W, 1), squeeze it. "
                    "If you have a volume (D, H, W), set is_volume=True."
                )
            # Promote 2D (H, W) -> 3D (1, H, W) for unified processing
            data = data[None, :, :]
        else:
            if data.ndim != 3:
                raise ValueError(
                    f"For 3D compression, input must be (D, H, W). Got {data.shape}"
                )
            # Data is already (D, H, W), keep it.

        # At this point, data is strictly (D, H, W)
        depth, height, width = data.shape

        # 2. Pad (Height and Width only)
        # ------------------------------
        pad_h = (4 - height % 4) % 4
        pad_w = (4 - width % 4) % 4

        if pad_h > 0 or pad_w > 0:
            # Pad H (axis 1) and W (axis 2). Do not pad Depth (axis 0).
            data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")

        d, h, w = data.shape

        # 3. Convert to float32
        # ---------------------
        if data.dtype != np.float32 and data.dtype != np.float16:
            # BC4 compression should be done on normalized float data (0.0 to 1.0) for best results.
            # Converting to uin8 beforehand would lose precision. So it is treated as an error.
            raise ValueError(
                f"Input data must be float16 or float32. Got {data.dtype}."
            )

        data_f32 = np.ascontiguousarray(data, dtype=np.float32)

        # 4. WebGPU Resources
        # -------------------
        # Note: WGPU texture size is (Width, Height, Depth)
        texture = self.device.create_texture(
            size=(w, h, d),
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            dimension="3d",  # Explicit 3D texture
            format="r32float",
            mip_level_count=1,
            sample_count=1,
        )

        # Upload data
        # Data layout matches texture size:
        # bytes_per_row = Width * 4
        # rows_per_image = Height
        self.device.queue.write_texture(
            {"texture": texture},
            data_f32,
            {"bytes_per_row": w * 4, "rows_per_image": h},
            (w, h, d),
        )

        # Output Buffer
        num_blocks = (w // 4) * (h // 4) * d
        output_size = num_blocks * 8

        buffer = self.device.create_buffer(
            size=output_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )

        bind_group = self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {"binding": 0, "resource": texture.create_view(dimension="3d")},
                {
                    "binding": 1,
                    "resource": {"buffer": buffer, "offset": 0, "size": output_size},
                },
            ],
        )

        # 5. Dispatch
        # -----------
        command_encoder = self.device.create_command_encoder()
        pass_encoder = command_encoder.begin_compute_pass()
        pass_encoder.set_pipeline(self.compute_pipeline)
        pass_encoder.set_bind_group(0, bind_group, [], 0, 99)

        # Dispatch (BlocksX, BlocksY, Depth)
        # Workgroup size is (8, 8, 1)
        dis_x = (w // 4 + 7) // 8
        dis_y = (h // 4 + 7) // 8
        dis_z = d  # One thread per Z slice roughly, but since Z group size is 1, it's exact.

        pass_encoder.dispatch_workgroups(dis_x, dis_y, dis_z)
        pass_encoder.end()

        self.device.queue.submit([command_encoder.finish()])

        return bytes(self.device.queue.read_buffer(buffer))

    @staticmethod
    def decompress(
        bc4_data: bytes, width: int, height: int, depth: int = 1
    ) -> np.ndarray:
        """Accelerated BC4 decompression using Numba."""
        # Ensure data is in a format Numba can read efficiently (uint8 array)
        bc4_arr = np.frombuffer(bc4_data, dtype=np.uint8)

        blocks_w = width // 4
        blocks_h = height // 4

        return _decompress_bc4_kernel(bc4_arr, depth, blocks_h, blocks_w, height, width)
