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
import argparse
from dwd_connect import DWDDownloader
from icon_loader import load_region
from datetime import datetime

# A region covering the Alps (good mix of thin and thick clouds)
REGION_CFG = {"lat_min": 46.2, "lat_max": 49.2, "lon_min": 9.4, "lon_max": 17.4}


def analyze_physics_limits(run_time, step):
    print(f"--- Fetching Data for {run_time} +{step} ---")
    downloader = DWDDownloader("analysis_cache")
    # We only need the physics variables
    files = downloader.fetch_variables(run_time, step, ["hhl", "qc", "clc"])
    cube = load_region(files, **REGION_CFG)

    print("--- Computing Theoretical Extinction Limits ---")

    # 1. Extract Data
    qc = cube.data["qc"]  # kg/kg
    clc = cube.data["clc"]  # 0..100
    hhl = cube.data["hhl"]  # meters

    # 2. Filter valid clouds
    # Ignore areas with < 1% cloud cover or negligible water
    mask = (clc > 1.0) & (qc > 1e-6)

    qc_valid = qc[mask]
    clc_valid = clc[mask]

    # 3. Apply your Shader Physics Formula
    # Convert clc 0..100 to fraction 0..1
    clc_frac = clc_valid / 100.0
    safe_clc = np.maximum(clc_frac, 0.01)

    # Estimate density (rho)
    # Ideally we'd use 'p' and 't' fields, but let's approximate based on height
    # Rho roughly 1.0 kg/m3 in lower atmosphere, 0.5 at 6km
    # Let's use 1.0 for a conservative (higher extinction) estimate
    rho = 1.0

    # In-cloud Liquid Water Content (g/m3)
    # qc is grid-averaged. We divide by clc to get the density *inside* the cloud chunk.
    # qc (kg/kg) * 1000 (g/kg) * rho (kg/m3) / clc_fraction
    c_liq = (qc_valid / safe_clc) * rho * 1000.0

    # Extinction Formula: beta = 140 * c_liq^0.7
    beta = 140.0 * np.power(c_liq, 0.7)

    # 4. Statistics
    p50 = np.percentile(beta, 50)
    p90 = np.percentile(beta, 90)
    p95 = np.percentile(beta, 95)
    p98 = np.percentile(beta, 98)
    p99 = np.percentile(beta, 99)
    p_max = np.max(beta)

    print(f"\n--- Extinction Statistics (km^-1) ---")
    print(f"Total Cloud Voxels: {len(beta)}")
    print(f"Median Cloud Density: {p50:.2f}")
    print(f"90th Percentile:      {p90:.2f}  (Dense)")
    print(f"95th Percentile:      {p95:.2f}  (Very Dense)")
    print(f"98th Percentile:      {p98:.2f}  (Storm Core)")
    print(f"99th Percentile:      {p99:.2f}  (Extreme)")
    print(f"Absolute Maximum:     {p_max:.2f}")

    print(f"\n--- Recommendation ---")
    print(f"To avoid banding (step < 0.2), MaxDensity should be < 50.")
    print(f"To avoid clamping cores, MaxDensity should be > {int(p98)}.")

    suggested = min(max(p98, 40), 80)  # Clamp suggestion between 40 and 80
    print(f"Recommended Linear MaxDensity: {int(suggested)}")
    print(f"This gives a precision step of: {suggested/255.0:.3f} km^-1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="YYYYMMDDHH")
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()

    dt = datetime.strptime(args.run, "%Y%m%d%H")
    analyze_physics_limits(dt, args.step)
