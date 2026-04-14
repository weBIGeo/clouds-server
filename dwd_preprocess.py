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

import numpy as np
import math
from numba import njit, prange


# --- Physics Helpers ---
@njit(inline="always")
def get_air_density(altitude_m):
    return 1.225 * math.exp(-altitude_m / 8400.0)


@njit(inline="always")
def get_extinction(qc, qi, rho_air):
    # OUTDATED!
    c_liq = qc * rho_air * 1000.0
    c_ice = qi * rho_air * 1000.0
    beta = 0.0
    if c_liq > 1e-6:
        beta += 140.0 * math.pow(c_liq, 0.7)
    if c_ice > 1e-6:
        beta += 60.0 * c_ice
    return beta


# --- Union-Find ---
@njit(inline="always")
def get_root(parent, i):
    root = i
    while root != parent[root]:
        root = parent[root]
    curr = i
    while curr != root:
        nxt = parent[curr]
        parent[curr] = root
        curr = nxt
    return root


@njit(inline="always")
def union_sets(parent, i, j):
    root_i = get_root(parent, i)
    root_j = get_root(parent, j)
    if root_i != root_j:
        parent[root_i] = root_j


@njit(parallel=True)
def clean_and_remap_clc(clc_array):
    nz, ny, nx = clc_array.shape
    out = np.zeros_like(clc_array)
    MIN_VAL = 0.001
    SCALE = 1.0 / (1.0 - MIN_VAL)
    for k in prange(nz):
        for j in range(ny):
            for i in range(nx):
                val = clc_array[k, j, i]
                if val < MIN_VAL:
                    out[k, j, i] = 0.0
                else:
                    out[k, j, i] = (val - MIN_VAL) * SCALE
    return out


@njit
def separate_cirrus_layers(clc, hhl, qc, qi):
    """
    Identifies detached high-altitude cloud islands using Mass Fraction analysis.
    Islands are flattened if:
    A) They are completely above SPLIT_ALTITUDE, OR
    B) >90% of their mass is above SPLIT_ALTITUDE.
    """
    nz, ny, nx = clc.shape
    num_voxels = nz * ny * nx
    parent = np.arange(num_voxels, dtype=np.int32)

    # 1. Connectivity Pass
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                val = clc[z, y, x]
                if val > 0.0:
                    idx = z * (ny * nx) + y * nx + x
                    # Union Neighbors
                    if x > 0 and clc[z, y, x - 1] > 0.0:
                        union_sets(parent, idx, idx - 1)
                    if y > 0 and clc[z, y - 1, x] > 0.0:
                        union_sets(parent, idx, idx - nx)
                    if z > 0 and clc[z - 1, y, x] > 0.0:
                        union_sets(parent, idx, idx - (ny * nx))

    # 2. Analysis Initialization
    # We track Mass (Sum of CLC) to determine if the island is "mostly" high up.
    root_mass_total = np.zeros(num_voxels, dtype=np.float32)
    root_mass_high = np.zeros(num_voxels, dtype=np.float32)

    # We still track altitude for layer assignment
    root_sum_h = np.zeros(num_voxels, dtype=np.float32)
    root_count = np.zeros(num_voxels, dtype=np.int32)
    root_min_z_idx = np.full(num_voxels, -1, dtype=np.int32)

    SPLIT_ALTITUDE = 7000.0  # The "Cirrus Line"
    CIRRUS_MASS_RATIO = 0.90  # 90% of cloud must be above line

    # 3. Analysis Accumulation
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                val = clc[z, y, x]
                if val > 0.0:
                    idx = z * (ny * nx) + y * nx + x
                    root = get_root(parent, idx)

                    # Mass Accumulation
                    root_mass_total[root] += val

                    # Altitude Check
                    # hhl[z] is top, hhl[z+1] is bottom
                    h_m = 0.5 * (hhl[z, y, x] + hhl[z + 1, y, x])

                    if h_m > SPLIT_ALTITUDE:
                        root_mass_high[root] += val

                    # Stats for Layer Assignment
                    root_sum_h[root] += h_m
                    root_count[root] += 1

                    # Deepest point tracking (Strict check backup)
                    if z > root_min_z_idx[root]:
                        root_min_z_idx[root] = z

    # 4. Flattening Pass
    cirrus_layers = np.zeros((3, ny, nx), dtype=np.float32)
    L1_TOP = 9000.0
    L2_TOP = 11000.0

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if clc[z, y, x] > 0.0:
                    idx = z * (ny * nx) + y * nx + x
                    root = get_root(parent, idx)

                    # --- CLASSIFICATION LOGIC ---
                    is_cirrus = False

                    # Criteria 1: Strict Altitude (Deepest point is above 7km)
                    deepest_idx = root_min_z_idx[root]
                    h_deepest = hhl[deepest_idx + 1, y, x]
                    if h_deepest > SPLIT_ALTITUDE:
                        is_cirrus = True

                    # Criteria 2: Mass Fraction (Soft Detachment)
                    # Even if a tail hangs down, if 90% is high, we take it.
                    elif root_mass_total[root] > 0.0:
                        ratio = root_mass_high[root] / root_mass_total[root]
                        if ratio > CIRRUS_MASS_RATIO:
                            is_cirrus = True

                    if is_cirrus:
                        # 1. Determine Layer
                        avg_h = root_sum_h[root] / root_count[root]
                        layer_idx = 0
                        if avg_h > L2_TOP:
                            layer_idx = 2
                        elif avg_h > L1_TOP:
                            layer_idx = 1

                        # 2. Flatten
                        h_m = 0.5 * (hhl[z, y, x] + hhl[z + 1, y, x])
                        dz = hhl[z, y, x] - hhl[z + 1, y, x]
                        rho = get_air_density(h_m)
                        beta = get_extinction(
                            qc[z, y, x], qi[z, y, x], rho
                        )

                        cirrus_layers[layer_idx, y, x] += beta * dz

                        # 3. Remove from 3D Volume
                        clc[z, y, x] = 0.0

    return cirrus_layers
