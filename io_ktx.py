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

import math
import struct
import json
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, TypedDict

import numpy as np

# Optional Compression Support
try:
    import zstandard as zstd

    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


class KtxFormatInfo(TypedDict):
    vk_id: int
    channels: int
    type_size: int
    dtype: Any
    dfd_flags: int
    is_float: bool
    is_signed: bool
    is_snorm: bool
    # Block Compression info
    block_dim: Optional[Tuple[int, int, int]]  # (w, h, d)
    block_bytes: Optional[int]  # Bytes per block
    dfd_model: Optional[int]  # for KHR_DF Color Model


class Ktx2:
    """
    A compliant KTX2 (v2.0) Reader/Writer for 3D Textures.

    Adheres to:
    - KTX File Format Specification Version 2.0
    - KHR_DF Data Format Specification 1.3

    Features:
    - Strict 16-byte alignment (via KVD padding) to pass 'ktx validate'.
    - Zstandard compression support.
    - Automatic DFD (Data Format Descriptor) generation.
    - Support for BC4 Block Compression.
    """

    # --- Constants ---
    IDENTIFIER = b"\xabKTX 20\xbb\r\n\x1a\n"
    HEADER_SIZE = 80
    LEVEL_INDEX_SIZE = 24

    # Base alignment for KTX2 (Level 0)
    BASE_ALIGNMENT = 16

    KTX_SS_NONE = 0
    KTX_SS_ZSTD = 2

    WRITER_ID_KEY = "KTXwriter"
    WRITER_ID_VAL = "Prototype KTX2"

    # --- Format Registry ---
    # fmt: off
    FORMATS: Dict[str, KtxFormatInfo] = {
        # Unsigned Normalized
        "R8_UNORM":         {"vk_id": 9,  "channels": 1, "type_size": 1, "dtype": np.uint8, "dfd_flags": 0x00, "is_float": False, "is_signed": False, "block_dim": None, "block_bytes": None, "dfd_model": 1},
        "R8G8_UNORM":       {"vk_id": 16, "channels": 2, "type_size": 1, "dtype": np.uint8, "dfd_flags": 0x00, "is_float": False, "is_signed": False, "block_dim": None, "block_bytes": None, "dfd_model": 1},
        "R8G8B8_UNORM":     {"vk_id": 23, "channels": 3, "type_size": 1, "dtype": np.uint8, "dfd_flags": 0x00, "is_float": False, "is_signed": False, "block_dim": None, "block_bytes": None, "dfd_model": 1},
        "R8G8B8A8_UNORM":   {"vk_id": 37, "channels": 4, "type_size": 1, "dtype": np.uint8, "dfd_flags": 0x00, "is_float": False, "is_signed": False, "block_dim": None, "block_bytes": None, "dfd_model": 1},

        # Signed Normalized
        "R8_SNORM":         {"vk_id": 10, "channels": 1, "type_size": 1, "dtype": np.int8,  "dfd_flags": 0x40, "is_float": False, "is_signed": True, "is_snorm": True, "block_dim": None, "block_bytes": None, "dfd_model": 1},
        "R8G8_SNORM":       {"vk_id": 17, "channels": 2, "type_size": 1, "dtype": np.int8,  "dfd_flags": 0x40, "is_float": False, "is_signed": True, "is_snorm": True, "block_dim": None, "block_bytes": None, "dfd_model": 1},
        "R8G8B8_SNORM":     {"vk_id": 24, "channels": 3, "type_size": 1, "dtype": np.int8,  "dfd_flags": 0x40, "is_float": False, "is_signed": True, "is_snorm": True, "block_dim": None, "block_bytes": None, "dfd_model": 1},
        "R8G8B8A8_SNORM":   {"vk_id": 38, "channels": 4, "type_size": 1, "dtype": np.int8,  "dfd_flags": 0x40, "is_float": False, "is_signed": True, "is_snorm": True, "block_dim": None, "block_bytes": None, "dfd_model": 1},

        # Unsigned Integer
        "R8_UINT":          {"vk_id": 13, "channels": 1, "type_size": 1, "dtype": np.uint8, "dfd_flags": 0x00, "is_float": False, "is_signed": False, "block_dim": None, "block_bytes": None, "dfd_model": 1},
        "R8G8_UINT":        {"vk_id": 20, "channels": 2, "type_size": 1, "dtype": np.uint8, "dfd_flags": 0x00, "is_float": False, "is_signed": False, "block_dim": None, "block_bytes": None, "dfd_model": 1},
        "R8G8B8_UINT":      {"vk_id": 27, "channels": 3, "type_size": 1, "dtype": np.uint8, "dfd_flags": 0x00, "is_float": False, "is_signed": False, "block_dim": None, "block_bytes": None, "dfd_model": 1},
        "R8G8B8A8_UINT":    {"vk_id": 41, "channels": 4, "type_size": 1, "dtype": np.uint8, "dfd_flags": 0x00, "is_float": False, "is_signed": False, "block_dim": None, "block_bytes": None, "dfd_model": 1},

        # Signed Integer
        "R8_SINT":          {"vk_id": 14, "channels": 1, "type_size": 1, "dtype": np.int8,  "dfd_flags": 0x40, "is_float": False, "is_signed": True, "block_dim": None, "block_bytes": None, "dfd_model": 1},
        "R8G8_SINT":        {"vk_id": 21, "channels": 2, "type_size": 1, "dtype": np.int8,  "dfd_flags": 0x40, "is_float": False, "is_signed": True, "block_dim": None, "block_bytes": None, "dfd_model": 1},
        "R8G8B8_SINT":      {"vk_id": 28, "channels": 3, "type_size": 1, "dtype": np.int8,  "dfd_flags": 0x40, "is_float": False, "is_signed": True, "block_dim": None, "block_bytes": None, "dfd_model": 1},
        "R8G8B8A8_SINT":    {"vk_id": 42, "channels": 4, "type_size": 1, "dtype": np.int8,  "dfd_flags": 0x40, "is_float": False, "is_signed": True, "block_dim": None, "block_bytes": None, "dfd_model": 1},

        # Float 16
        "R16_SFLOAT":       {"vk_id": 76, "channels": 1, "type_size": 2, "dtype": np.float16, "dfd_flags": 0xC0, "is_float": True, "is_signed": True, "block_dim": None, "block_bytes": None, "dfd_model": 1},
        "R16G16_SFLOAT":    {"vk_id": 83, "channels": 2, "type_size": 2, "dtype": np.float16, "dfd_flags": 0xC0, "is_float": True, "is_signed": True, "block_dim": None, "block_bytes": None, "dfd_model": 1},
        "R16G16B16_SFLOAT": {"vk_id": 90, "channels": 3, "type_size": 2, "dtype": np.float16, "dfd_flags": 0xC0, "is_float": True, "is_signed": True, "block_dim": None, "block_bytes": None, "dfd_model": 1},
        "R16G16B16A16_SFLOAT": {"vk_id": 97, "channels": 4, "type_size": 2, "dtype": np.float16, "dfd_flags": 0xC0, "is_float": True, "is_signed": True, "block_dim": None, "block_bytes": None, "dfd_model": 1},

        # Float 32
        "R32_SFLOAT":       {"vk_id": 100, "channels": 1, "type_size": 4, "dtype": np.float32, "dfd_flags": 0xC0, "is_float": True, "is_signed": True, "block_dim": None, "block_bytes": None, "dfd_model": 1},
        "R32G32_SFLOAT":    {"vk_id": 103, "channels": 2, "type_size": 4, "dtype": np.float32, "dfd_flags": 0xC0, "is_float": True, "is_signed": True, "block_dim": None, "block_bytes": None, "dfd_model": 1},
        "R32G32B32_SFLOAT": {"vk_id": 106, "channels": 3, "type_size": 4, "dtype": np.float32, "dfd_flags": 0xC0, "is_float": True, "is_signed": True, "block_dim": None, "block_bytes": None, "dfd_model": 1},
        "R32G32B32A32_SFLOAT": {"vk_id": 109, "channels": 4, "type_size": 4, "dtype": np.float32, "dfd_flags": 0xC0, "is_float": True, "is_signed": True, "block_dim": None, "block_bytes": None, "dfd_model": 1},

        # BC4 Block Compressed
        "BC4_UNORM_BLOCK":  {"vk_id": 139, "channels": 1, "type_size": 1, "dtype": np.uint8, "dfd_flags": 0x00, "is_float": False, "is_signed": False, "block_dim": (4, 4, 1), "block_bytes": 8, "dfd_model": 131},
        "BC4_SNORM_BLOCK":  {"vk_id": 140, "channels": 1, "type_size": 1, "dtype": np.int8,  "dfd_flags": 0x40, "is_float": False, "is_signed": True,  "block_dim": (4, 4, 1), "block_bytes": 8, "dfd_model": 131},
    }
    # fmt: on

    VK_ID_TO_NAME = {v["vk_id"]: k for k, v in FORMATS.items()}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @classmethod
    def save(
        cls,
        data: Union[np.ndarray, list[np.ndarray]],
        path: Union[str, Path],
        target_format: str,
        dimension: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
        compress: bool = False,
        compression_level: int = 3,
        shape: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Saves a 3D texture (or list of mip levels) to a KTX2 file.

        Args:
            data: Single numpy array (Level 0) OR List of numpy arrays (Level 0..N).
                  For Block Compressed formats, these must be raw bytes/uint8 1D arrays.
            path: Destination file path.
            target_format: Key from Ktx2.FORMATS.
            metadata: Dictionary of key-value pairs.
            compress: If True, uses Zstandard compression.
            dimension: 2 or 3.
                - If 2: Input Data must be (H, W, C). Header pixelDepth will be 0.
                - If 3: Input Data must be (D, H, W, C). Header pixelDepth will be D.
            shape: Explicit shape required ONLY if `data` is raw bytes (Block Compressed).
                   - If dimension=2: (H, W)
                   - If dimension=3: (D, H, W)
        """
        if dimension not in (2, 3):
            raise ValueError("dimension must be 2 or 3")

        if target_format not in cls.FORMATS:
            raise ValueError(f"Unknown format: {target_format}")

        fmt_info = cls.FORMATS[target_format]
        is_block_fmt = fmt_info.get("block_bytes") is not None
        levels = data if isinstance(data, list) else [data]

        num_levels = len(levels)
        if num_levels == 0:
            raise ValueError("Data list is empty.")

        lvl0 = levels[0]

        if is_block_fmt:
            # Block Compressed need explicit shape arg
            if shape is None:
                raise ValueError("Shape required for raw bytes.")
            if dimension == 2:
                if len(shape) != 2:
                    raise ValueError("For 2D raw data, shape must be (H, W).")
                h, w = shape
                d = 1
                header_d = 0  # Spec: 0 for 2D
            else:
                if len(shape) != 3:
                    raise ValueError("For 3D raw data, shape must be (D, H, W).")
                d, h, w = shape
                header_d = d
        else:
            # Numpy Arrays - Let _prepare_data handle strict validation
            if dimension == 2:
                # Expect (H, W, C)
                h, w = lvl0.shape[0], lvl0.shape[1]
                d = 1
                header_d = 0
            else:
                # Expect (D, H, W, C)
                d, h, w = lvl0.shape[0], lvl0.shape[1], lvl0.shape[2]
                header_d = d

        # 3. Process Levels (Validate, Compress, Prepare)
        processed_levels = []

        for i, lvl_data in enumerate(levels):
            # Calc dimensions for this level
            cw = cls._get_mip_dim(w, i)
            ch = cls._get_mip_dim(h, i)
            cd = cls._get_mip_dim(d, i)

            # Prepare/Validate
            if is_block_fmt:
                arr_contig = cls._prepare_bc_data(lvl_data, cw, ch, cd, fmt_info)
            else:
                arr_contig = cls._prepare_data(
                    lvl_data, fmt_info, target_format, dimension
                )

            raw_bytes = arr_contig.tobytes(order="C")

            # Compress
            comp_bytes, scheme = cls._compress(raw_bytes, compress, compression_level)
            processed_levels.append(
                {
                    "data": comp_bytes,
                    "len": len(comp_bytes),
                    "uncomp_len": len(raw_bytes),
                }
            )

        # 4. Alignment & Layout Calculation

        # A. Determine Alignment Requirements
        if compress:
            # Spec: required_alignment = 1 if supercompressionScheme != 0 (Zstd)
            # This ensures tight packing between mip levels.
            level_align = 1
        else:
            # Spec: required_alignment = lcm(texel_block_size, 4) if Scheme == 0
            if is_block_fmt:
                bs = fmt_info["block_bytes"]
                level_align = cls._lcm(bs, 4)
            else:
                texel_sz = fmt_info["type_size"] * fmt_info["channels"]
                level_align = cls._lcm(texel_sz, 4)

        # B. Determine Start Alignment
        # The START of the image data (Level N) must be LCM(16, level_align).
        # Even if level_align is 1, the file structure requires 16-byte alignment here.
        start_align = cls._lcm(cls.BASE_ALIGNMENT, level_align)

        # Generate Metadata Headers
        dfd_bytes = cls._create_dfd(fmt_info)

        # Offsets
        off_dfd = cls.HEADER_SIZE + (cls.LEVEL_INDEX_SIZE * num_levels)
        off_kvd = off_dfd + len(dfd_bytes)

        # Tentative KVD
        temp_kvd = cls._create_kvd(metadata, extra_padding=0)
        end_kvd_tentative = off_kvd + len(temp_kvd)

        # Padding logic:
        # Instead of inserting 0s into the file, we extend the KVD data itself.
        # This ensures the KVD ends exactly where the aligned image data begins.
        align_gap = (start_align - (end_kvd_tentative % start_align)) % start_align

        if align_gap > 0:
            kvd_bytes = cls._create_kvd(metadata, extra_padding=align_gap)
        else:
            kvd_bytes = temp_kvd

        current_file_offset = off_kvd + len(kvd_bytes)

        # 5. Determine offsets for each level (Write Order: N -> 0)
        level_indices = [None] * num_levels

        for i in range(num_levels - 1, -1, -1):
            pl = processed_levels[i]

            # Determine alignment for THIS level chunk.
            # The FIRST physical chunk (i == N-1) relies on the start_align logic handled by KVD.
            # Subsequent chunks (N-2...0) use level_align (which is 1 for compressed).

            if i == num_levels - 1:
                # We already aligned current_file_offset via KVD padding above.
                req = start_align
            else:
                req = level_align

            # Calculate padding (should be 0 for compressed levels N-2..0)
            padding_needed = (req - (current_file_offset % req)) % req
            current_file_offset += padding_needed

            # Record offset
            offset = current_file_offset
            length = pl["len"]
            uncomp = pl["uncomp_len"]

            level_indices[i] = (offset, length, uncomp)

            # Advance cursor
            current_file_offset += length

        # 6. Header Construction
        header = bytearray(cls.IDENTIFIER)
        hdr_type_size = 1 if is_block_fmt else fmt_info["type_size"]

        # fmt: off
        header.extend(struct.pack("<IIIIIIIII", 
            fmt_info["vk_id"], 
            hdr_type_size, 
            w, h, header_d,
            0, 1, num_levels, # arrayCount, faceCount, levelCount
            scheme
        ))
        header.extend(struct.pack("<IIII", off_dfd, len(dfd_bytes), off_kvd, len(kvd_bytes)))
        
        # sgdByteOffset must be 0 if sgdByteLength is 0
        header.extend(struct.pack("<QQ", 0, 0)) 
        # fmt: on

        # Construct Index Block (Index Order: 0 -> N)
        index_block = bytearray()
        for i in range(num_levels):
            off, length, uncomp = level_indices[i]
            index_block.extend(struct.pack("<QQQ", off, length, uncomp))

        # 7. Write File
        with open(path, "wb") as f:
            f.write(header)
            f.write(index_block)
            f.write(dfd_bytes)
            f.write(kvd_bytes)

            # Write Data Payloads (Physical Order: N -> 0)
            f_pos = f.tell()

            for i in range(num_levels - 1, -1, -1):
                target_off = level_indices[i][0]
                padding = target_off - f_pos
                if padding > 0:
                    f.write(b"\x00" * padding)

                f.write(processed_levels[i]["data"])
                f_pos = f.tell()

    @classmethod
    def load(cls, path: Union[str, Path]) -> Tuple[list[np.ndarray], Dict[str, Any]]:
        """
        Loads a KTX2 file.

        Returns:
            Tuple(List of Numpy Arrays [Level0, Level1...], Metadata Dict)
        """
        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")

        with open(path, "rb") as f:
            # 1. Parse Header
            if f.read(12) != cls.IDENTIFIER:
                raise ValueError(f"File {path} is not a valid KTX2 file.")

            header_data = f.read(68)
            (
                vk_fmt,
                type_size,
                w,
                h,
                d,
                layers,
                faces,
                levels,
                scheme,
                dfd_off,
                dfd_len,
                kvd_off,
                kvd_len,
                sgd_off,
                sgd_len,
            ) = struct.unpack("<IIIIIIIII IIII QQ", header_data)

            calc_d = max(1, d)

            if vk_fmt not in cls.VK_ID_TO_NAME:
                raise ValueError(f"Unsupported Vulkan Format ID: {vk_fmt}")

            fmt_name = cls.VK_ID_TO_NAME[vk_fmt]
            fmt_info = cls.FORMATS[fmt_name]

            # 2. Parse Level Index (24 bytes per level)
            level_index = []
            for _ in range(levels):
                level_index.append(struct.unpack("<QQQ", f.read(24)))

            # 3. Parse Metadata
            metadata = cls._parse_kvd(f, kvd_off, kvd_len)
            metadata["_ktx_format"] = fmt_name
            metadata["_ktx_block_dimensions"] = (
                (w, h, calc_d) if fmt_info.get("block_bytes") else None
            )

            # 4. Read Payloads
            output_levels = []

            for i in range(levels):
                lvl_off, lvl_len, lvl_uncomp_len = level_index[i]

                f.seek(lvl_off)
                payload = f.read(lvl_len)
                if len(payload) != lvl_len:
                    raise IOError(f"Unexpected EOF reading Level {i}.")

                # Decompress
                if scheme == cls.KTX_SS_ZSTD:
                    if not HAS_ZSTD:
                        raise RuntimeError("Zstd required but not installed.")
                    data_bytes = zstd.ZstdDecompressor().decompress(
                        payload, max_output_size=lvl_uncomp_len
                    )
                elif scheme == cls.KTX_SS_NONE:
                    data_bytes = payload
                else:
                    raise ValueError(f"Unsupported Scheme: {scheme}")

                # Reconstruct Array
                if fmt_info.get("block_bytes"):
                    # Raw bytes for blocks
                    arr = np.frombuffer(data_bytes, dtype=np.uint8)
                else:
                    arr = np.frombuffer(data_bytes, dtype=fmt_info["dtype"])

                    # Calculate dimensions for this level
                    cw = cls._get_mip_dim(w, i)
                    ch = cls._get_mip_dim(h, i)
                    cd = cls._get_mip_dim(d, i)
                    c = fmt_info["channels"]

                    if d == 0:
                        # (H, W, C)
                        arr = arr.reshape((ch, cw, c))
                    else:
                        # (D, H, W, C)
                        arr = arr.reshape((cd, ch, cw, c))
                output_levels.append(arr)

            return output_levels, metadata

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _lcm(a: int, b: int) -> int:
        """Computes Least Common Multiple."""
        return abs(a * b) // math.gcd(a, b)

    @staticmethod
    def _prepare_bc_data(
        data: np.ndarray, w: int, h: int, d: int, fmt: KtxFormatInfo
    ) -> np.ndarray:
        """Validates raw payload size for Block Compressed formats."""
        bw, bh, bd = fmt["block_dim"] or (1, 1, 1)
        b_bytes = fmt["block_bytes"] or 1

        # Calculate number of blocks (Ceiling division)
        blocks_w = (w + bw - 1) // bw
        blocks_h = (h + bh - 1) // bh
        blocks_d = (d + bd - 1) // bd

        expected_size = blocks_w * blocks_h * blocks_d * b_bytes

        # Convert bytes/bytearray to numpy array immediately to access .nbytes logic generically
        if isinstance(data, (bytes, bytearray)):
            arr = np.frombuffer(data, dtype=np.uint8)
        else:
            arr = data

        if arr.nbytes != expected_size:
            raise ValueError(
                f"Block Compressed Data Size Mismatch.\n"
                f"Dimensions: {w}x{h}x{d}\n"
                f"Block Count: {blocks_w}x{blocks_h}x{blocks_d}\n"
                f"Expected Bytes: {expected_size}, Got: {data.nbytes}"
            )

        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        return arr.view(np.uint8).flatten()

    @classmethod
    def _prepare_data(
        cls, data: np.ndarray, fmt: KtxFormatInfo, fmt_name: str, dimension: int
    ) -> np.ndarray:
        """Validates shape, casts type, and ensures C-contiguity."""
        expected_channels = fmt["channels"]
        arr = data

        if dimension == 2:
            # 2D Requirement: (H, W, C) -> ndim must be 3
            if arr.ndim != 3:
                raise ValueError(
                    f"For 2D, input must be (H, W, C). Got ndim={arr.ndim}"
                )
            if arr.shape[2] != expected_channels:
                raise ValueError(
                    f"Channel mismatch: Format {fmt_name} expects {expected_channels}, got {arr.shape[2]}"
                )
        elif dimension == 3:
            # 3D Requirement: (D, H, W, C) -> ndim must be 4
            if arr.ndim != 4:
                raise ValueError(
                    f"For 3D, input must be (D, H, W, C). Got ndim={arr.ndim}"
                )
            if arr.shape[3] != expected_channels:
                raise ValueError(
                    f"Channel mismatch: Format {fmt_name} expects {expected_channels}, got {arr.shape[3]}"
                )
        else:
            raise ValueError("dimension must be 2 or 3")

        # Range normalization / Type casting
        if np.issubdtype(data.dtype, np.floating) and not fmt["is_float"]:
            if "UNORM" in fmt_name:
                arr = np.clip(arr, 0, 1) * 255
            elif "SNORM" in fmt_name:
                arr = np.clip(arr, -1, 1) * 127

        # Final cast
        if arr.dtype != fmt["dtype"]:
            arr = arr.astype(fmt["dtype"])

        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        return arr

    @classmethod
    def _compress(
        cls, raw_data: bytes, compress: bool, level: int
    ) -> Tuple[bytes, int]:
        """Handles compression logic."""
        if compress:
            if HAS_ZSTD:
                return (
                    zstd.ZstdCompressor(level=level).compress(raw_data),
                    cls.KTX_SS_ZSTD,
                )
            else:
                print("Warning: 'zstandard' not installed. Saving uncompressed.")
                return raw_data, cls.KTX_SS_NONE
        return raw_data, cls.KTX_SS_NONE

    @staticmethod
    def _create_dfd(fmt_info: KtxFormatInfo) -> bytes:
        """
        Creates a DFD block compliant with KHR_DF 1.3.
        """
        channels = fmt_info["channels"]

        # Block Dimensions: Fix "Non-iterable NoneType" error
        bw, bh, bd = fmt_info["block_dim"] or (1, 1, 1)

        # Bytes Plane
        block_bytes = fmt_info.get("block_bytes")
        if block_bytes:
            bytes_plane = block_bytes
            # For block compressed, we write 1 sample describing the whole block
            num_samples = 1
        else:
            bytes_plane = fmt_info["channels"] * fmt_info["type_size"]
            num_samples = fmt_info["channels"]

        # Color Model: Default to RGBSDA (1) if not specified
        color_model = fmt_info.get("dfd_model", 1)

        # Fixed sizes
        HEADER_SZ = 8
        BODY_SZ = 16
        SAMPLE_SZ = 16

        # Spec: descriptorBlockSize includes the header
        block_size = HEADER_SZ + BODY_SZ + (channels * SAMPLE_SZ)

        # 1. Header
        # Vendor=0 (KHR), Type=0 (Basic), Version=2 (KHR_DF 1.3)
        dfd_header = struct.pack("<I H H", 0, 2, block_size)

        # 2. Basic Descriptor Body
        # KHR_DF 1.3 Spec:
        # - model: 1 (RGBSDA)
        # - primaries: 1 (BT.709)
        # - transfer: 1 (Linear)
        # - flags: 0 (Alpha Straight)
        # - texelBlockDimension[0..3]: Actual dimension minus 1
        # - bytesPlane[0..7]: Number of bytes in the block

        basic_desc = struct.pack(
            "<BBBB BBBB Q",
            color_model,  # model (1=RGBSDA, 131=BC4, etc.)
            1,  # primaries (BT.709)
            1,  # transfer (Linear)
            0,  # flags
            bw - 1,  # texelBlockDimension0 (x)
            bh - 1,  # texelBlockDimension1 (y)
            bd - 1,  # texelBlockDimension2 (z)
            0,  # texelBlockDimension3 (w)
            bytes_plane,  # bytesPlane0-7
        )

        # 3. Samples
        samples = bytearray()

        # Logic for Block Compressed (BC) vs Standard
        if block_bytes:
            # Block Compressed: 1 sample, spanning the full block size (64 or 128 bits)
            bit_offset = 0
            bit_len = (block_bytes * 8) - 1

            # For 64-bit blocks (BC4/BC1), lower=0, upper=UINT_MAX (since it exceeds 32 bits)
            lower, upper = 0, 0xFFFFFFFF

            # Channel Type: 0 (Unspecified/Red depending on interpretation, usually 0 for blocks)
            # DFD Flags usually 0 for the block sample wrapper
            c_type = 0

            sample_head = struct.pack(
                "<H B B 4B", bit_offset, bit_len, c_type, 0, 0, 0, 0
            )
            samples.extend(sample_head + struct.pack("<II", lower, upper))

        else:
            # Standard Uncompressed: 1 sample per channel

            # Bounds
            if fmt_info["is_float"]:
                p_lower, p_upper = struct.pack("<f", -1.0), struct.pack("<f", 1.0)
            elif fmt_info["is_signed"]:
                bits = fmt_info["type_size"] * 8

                # SNORM uses symmetric range [-(2^(n-1)-1), 2^(n-1)-1], e.g. [-127, 127] for 8-bit.
                # SINT uses full range [-128, 127].
                if fmt_info.get("is_snorm"):
                    min_v = -((1 << (bits - 1)) - 1)  # -127
                else:
                    min_v = -(1 << (bits - 1))  # -128

                max_v = (1 << (bits - 1)) - 1
                p_lower, p_upper = struct.pack("<i", min_v), struct.pack("<i", max_v)
            else:
                max_v = (1 << (fmt_info["type_size"] * 8)) - 1
                p_lower, p_upper = struct.pack("<I", 0), struct.pack("<I", max_v)

            bit_len = (fmt_info["type_size"] * 8) - 1
            channel_ids = [0, 1, 2, 15]  # R, G, B, A mapping

            for c in range(num_samples):
                bit_offset = c * (fmt_info["type_size"] * 8)
                cid = channel_ids[c] if c < 4 else 0
                c_type = fmt_info["dfd_flags"] | cid

                sample_head = struct.pack(
                    "<H B B 4B", bit_offset, bit_len, c_type, 0, 0, 0, 0
                )
                samples.extend(sample_head + p_lower + p_upper)

        content = dfd_header + basic_desc + samples
        return struct.pack("<I", len(content) + 4) + content

    @classmethod
    def _create_kvd(
        cls, metadata: Optional[Dict[str, Any]], extra_padding: int = 0
    ) -> bytes:
        """
        Creates the Key/Value Data block.
        Appends underscores to the writer signature to force 16-byte alignment of the *next* block.
        """
        kv = {cls.WRITER_ID_KEY: cls.WRITER_ID_VAL}
        if metadata:
            kv.update(metadata)

        # Inject padding into the Writer ID value
        if extra_padding > 0:
            kv[cls.WRITER_ID_KEY] += "_" * extra_padding

        buf = bytearray()
        for k, v in sorted(kv.items()):
            if not isinstance(v, str):
                v = str(v)

            k_bytes = k.encode("utf-8") + b"\x00"
            v_bytes = v.encode("utf-8") + b"\x00"

            # KTX2 Spec: Entries are 4-byte aligned
            entry_len = len(k_bytes) + len(v_bytes)
            pad_len = (4 - (entry_len % 4)) % 4

            buf.extend(
                struct.pack("<I", entry_len) + k_bytes + v_bytes + (b"\x00" * pad_len)
            )

        return bytes(buf)

    @staticmethod
    def _parse_kvd(f, offset: int, length: int) -> Dict[str, Any]:
        """Parses the Key/Value Data block safely."""
        metadata = {}
        if length == 0:
            return metadata

        f.seek(offset)
        blob = f.read(length)
        ptr = 0

        while ptr < len(blob):
            if ptr + 4 > len(blob):
                break

            kv_len = struct.unpack("<I", blob[ptr : ptr + 4])[0]
            ptr += 4

            if ptr + kv_len > len(blob):
                break

            kv_pair = blob[ptr : ptr + kv_len]
            split = kv_pair.find(b"\x00")

            if split != -1:
                key = kv_pair[:split].decode("utf-8", errors="ignore")
                # Value starts after null terminator
                val_bytes = kv_pair[split + 1 :]
                # Strip value null terminator if present
                if val_bytes and val_bytes[-1] == 0:
                    val_bytes = val_bytes[:-1]

                val_str = val_bytes.decode("utf-8", errors="ignore")

                # Attempt JSON decoding for complex metadata
                if key == "metadata":
                    try:
                        metadata.update(json.loads(val_str))
                    except json.JSONDecodeError:
                        metadata[key] = val_str
                else:
                    metadata[key] = val_str

            # Skip alignment padding
            pad = (4 - (kv_len % 4)) % 4
            ptr += kv_len + pad

        return metadata

    @staticmethod
    def _get_mip_dim(base: int, level: int) -> int:
        """Calculates dimension for a specific mip level (floored at 1)."""
        return max(1, base >> level)


# -------------------------------------------------------------------------
# TESTS
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- KTX2 I/O Test ---")

    # 1. Standard Uncompressed Test
    D, H, W = 16, 64, 64
    vol_orig = np.random.rand(D, H, W).astype(np.float32)[:, :, :, np.newaxis]

    print("Saving R32_SFLOAT (Zstd)...")
    Ktx2.save(
        vol_orig,
        "io_test_r32.ktx2",
        "R32_SFLOAT",
        compress=True,
        metadata={"Author": "Me"},
    )

    # 2. BC4 Compression Test
    print("\n--- BC4 Block Compression Test ---")
    try:
        from bc4 import BC4Compressor

        # Create a test volume for BC4 (Single Channel)
        # BC4 works on 4x4 blocks. Let's make dimensions multiples of 4 for simplicity.
        bd, bh, bw = 8, 32, 32

        # Create a gradient pattern
        vol_bc4 = np.zeros((bd, bh, bw), dtype=np.uint8)
        for z in range(bd):
            for y in range(bh):
                for x in range(bw):
                    vol_bc4[z, y, x] = (x + y + z * 4) % 255

        print(f"Original Volume Shape: {vol_bc4.shape}")

        # Compress
        compressor = BC4Compressor()
        print("Compressing using BC4Compressor...")
        bc4_payload = compressor.compress(vol_bc4, is_volume=True)
        print(f"Compressed Payload Size: {len(bc4_payload)} bytes")

        # Verify size logic:
        # (32/4) * (32/4) * (8/1) = 8 * 8 * 8 = 512 blocks.
        # 512 blocks * 8 bytes/block = 4096 bytes.
        expected_bytes = (bw // 4) * (bh // 4) * bd * 8
        if len(bc4_payload) != expected_bytes:
            print(
                f"WARNING: Payload size {len(bc4_payload)} != Expected {expected_bytes}"
            )

        # Save as KTX2
        # Note: Must pass 'shape' for block compressed data
        print("Saving BC4_UNORM_BLOCK KTX2...")
        Ktx2.save(
            bc4_payload,
            "io_test_bc4.ktx2",
            "BC4_UNORM_BLOCK",
            shape=vol_bc4.shape,
            metadata={"Compression": "BC4"},
        )

        # Load back
        # Note: load() returns the raw compressed bytes for block formats
        print("Loading back BC4 KTX2...")
        mips, meta = Ktx2.load("io_test_bc4.ktx2")
        loaded_bytes = mips[0]

        # Verify
        if np.array_equal(
            loaded_bytes.flatten(), np.frombuffer(bc4_payload, dtype=np.uint8)
        ):
            print("SUCCESS: BC4 payload matches exactly.")
        else:
            print("FAILURE: BC4 payload mismatch.")

    except ImportError:
        print("Skipping BC4 tests: 'bc4' module not found.")
    except Exception as e:
        print(f"FAILURE during BC4 test: {e}")
        import traceback

        traceback.print_exc()

        # 1. Create and Save (Compressed)
    D, H, W = 16, 64, 64
    vol_orig = np.random.rand(D, H, W).astype(np.float32)[:, :, :, np.newaxis]

    print("Saving R32_SFLOAT (Zstd)...")
    Ktx2.save(
        vol_orig,
        "io_test_r32.ktx2",
        "R32_SFLOAT",
        compress=True,
        metadata={"Author": "Me"},
    )

    # 2. Load
    print("Loading back...")
    mips, meta = Ktx2.load("io_test_r32.ktx2")
    vol_loaded = mips[0]

    # 3. Verify
    print(f"Loaded Shape: {vol_loaded.shape}")
    print(f"Loaded Metadata: {meta}")

    if np.allclose(vol_orig, vol_loaded):
        print("SUCCESS: Data matches exactly.")
    else:
        print("FAILURE: Data mismatch.")

    # 4. Test Multi-Channel Read/Write
    print("\nSaving R8G8 (UV Ramp)...")
    vol_uv = np.zeros((D, H, W, 2), dtype=np.uint8)
    vol_uv[..., 0] = np.linspace(0, 255, W).astype(np.uint8)[np.newaxis, np.newaxis, :]
    vol_uv[..., 1] = np.linspace(0, 255, H).astype(np.uint8)[np.newaxis, :, np.newaxis]

    Ktx2.save(vol_uv, "io_test_uv.ktx2", "R8G8_UNORM")
    mips, _ = Ktx2.load("io_test_uv.ktx2")
    vol_uv_loaded = mips[0]

    if np.array_equal(vol_uv, vol_uv_loaded):
        print("SUCCESS: Multi-channel data matches.")
    else:
        print("FAILURE: Multi-channel mismatch.")

    # Dimensions (Z, Y, X)
    D, H, W = 140, 1088, 1088
    print(f"--- Test: RGB8 UVW Volume ({W}x{H}x{D}) ---")

    # 1. Allocate Array (using uint8 directly to save memory)
    # Shape: (Z, Y, X, 3)
    vol = np.zeros((D, H, W, 3), dtype=np.uint8)

    print("Generating UVW gradients...")
    # 2. Fill Channels using Broadcasting
    # Red   = X (U) 0..255
    # Green = Y (V) 0..255
    # Blue  = Z (W) 0..255

    # Create 1D ramps and broadcast them to 3D
    x_ramp = np.linspace(0, 255, W).astype(np.uint8)
    y_ramp = np.linspace(0, 255, H).astype(np.uint8)
    z_ramp = np.linspace(0, 255, D).astype(np.uint8)

    vol[..., 0] = x_ramp[np.newaxis, np.newaxis, :]  # Broadcast along Z, Y
    vol[..., 1] = y_ramp[np.newaxis, :, np.newaxis]  # Broadcast along Z, X
    vol[..., 2] = z_ramp[:, np.newaxis, np.newaxis]  # Broadcast along Y, X

    # 3. Save
    filename = "io_test_uvw_large.ktx2"
    print(f"Saving {filename}...")
    Ktx2.save(vol, filename, "R8G8B8_UNORM")

    # 4. Load & Verify
    print("Loading back for verification...")
    mips, meta = Ktx2.load(filename)
    vol_loaded = mips[0]

    print(f"Loaded Shape: {vol_loaded.shape}")

    if np.array_equal(vol, vol_loaded):
        print("SUCCESS: Volume data matches exactly.")
    else:
        print("FAILURE: Data mismatch.")

    D, H, W = 140, 1088, 1088
    print(f"--- Test: R8 Large Volume ({W}x{H}x{D}) ---")

    # 1. Generate Data
    # Pattern: Diagonal gradient ((x + y + z) % 255)
    # We use uint16 for the math to prevent premature overflow, then cast to uint8
    print("Generating procedural volumetric data...")

    # Create 1D coordinate arrays
    z_coords = np.arange(D, dtype=np.uint16).reshape(D, 1, 1)
    y_coords = np.arange(H, dtype=np.uint16).reshape(1, H, 1)
    x_coords = np.arange(W, dtype=np.uint16).reshape(1, 1, W)

    # Broadcast add: result is (D, H, W)
    # The % 255 creates a repeating saw-tooth wave pattern through the volume
    vol = ((z_coords + y_coords + x_coords) % 255).astype(np.uint8)[:, :, :, np.newaxis]

    print(f"Volume Size: {vol.nbytes / 1024 / 1024:.2f} MB")

    # 2. Save
    filename = "io_test_r8_large.ktx2"
    print(f"Saving {filename}...")
    Ktx2.save(vol, filename, "R8_UNORM")

    # 3. Load & Verify
    print("Loading back for verification...")
    mips, meta = Ktx2.load(filename)
    vol_loaded = mips[0]

    print(f"Loaded Shape: {vol_loaded.shape}")

    if np.array_equal(vol, vol_loaded):
        print("SUCCESS: Volume data matches exactly.")
    else:
        print("FAILURE: Data mismatch.")

    print("\n--- Test: Mipmaps with Alignment Check ---")

    # Define format and dimensions
    fmt = "R8G8B8A8_UNORM"
    D, H, W = 1, 64, 64
    filename_mips = "io_test_mips_aligned.ktx2"

    # Create 3 Levels
    # Level 0: 64x64 - Red
    l0 = np.zeros((D, H, W, 4), dtype=np.uint8)
    l0[:] = [255, 0, 0, 255]

    # Level 1: 32x32 - Green
    l1 = np.zeros((D, H // 2, W // 2, 4), dtype=np.uint8)
    l1[:] = [0, 255, 0, 255]

    # Level 2: 16x16 - Blue
    l2 = np.zeros((D, H // 4, W // 4, 4), dtype=np.uint8)
    l2[:] = [0, 0, 255, 255]

    levels_input = [l0, l1, l2]

    # Save
    print(f"Saving {len(levels_input)} levels to {filename_mips}...")
    Ktx2.save(
        levels_input, filename_mips, fmt, metadata={"Content": "Mipmap Test Colors"}
    )

    # Verification Instruction
    print(f"File saved. Run the following command to verify compliance:")
    print(f"  ktx validate {filename_mips}")

    # Load back to check Python integrity
    loaded_levels, meta = Ktx2.load(filename_mips)
    print(f"Loaded back {len(loaded_levels)} levels.")

    # Check dimensions and colors
    success = True
    if loaded_levels[0].shape != (1, 64, 64, 4):
        print(f"Error: L0 shape mismatch. Got {loaded_levels[0].shape}")
        success = False

    if loaded_levels[1].shape != (1, 32, 32, 4):
        print(f"Error: L1 shape mismatch. Got {loaded_levels[1].shape}")
        success = False

    # Check color sampling (center pixel)
    if not np.allclose(loaded_levels[1][0, 16, 16], [0, 255, 0, 255]):
        print(f"Error: L1 color mismatch. Expected Green.")
        success = False

    if success:
        print("SUCCESS: Python load verification passed.")

    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # Red Channel

    filename = "io_test_2d.ktx2"

    print(f"Saving 2D texture to {filename}...")

    # 2. Save with strict dimension=2
    Ktx2.save(img, filename, "R8G8B8_UNORM", dimension=2)  # <--- Explicit Requirement

    loaded_levels, meta = Ktx2.load(filename)
    if loaded_levels[0].shape == (128, 128, 3):
        print("SUCCESS: 2D texture saved and loaded correctly.")
    else:
        print(f"FAILURE: Loaded shape mismatch: {loaded_levels[0].shape}")
