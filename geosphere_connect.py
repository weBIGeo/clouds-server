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

import json
import logging
import os
import sys

import requests

logger = logging.getLogger("geosphere_connect")

# =============================================================================
# Dataset: SNOWGRID-CL
#  Overview: https://data.hub.geosphere.at/dataset/snowgrid_cl-v2-1d-1km
#  API base: https://dataset.api.hub.geosphere.at/v1/grid/historical/snowgrid_cl-v2-1d-1km
#  Coverage: Austria, 1961-01-01 to present (published with ~1-2 day lag)
#  Variables: snow_depth (m), swe_tot (kg m-2)
#  Grid: ~192 136 points at 1 km resolution (EPSG:3416)
#
# Rate limits (https://dataset.api.hub.geosphere.at/v1/docs/user-guide/request-limit.html):
#  - 5 requests per second, 240 requests per hour
#  - Max values per request: 1000000 (JSON/CSV) or 10 000000 (NetCDF)
#    Calculated as: parameters × time steps × grid points
#    NOTE: here its 2 params * 1 day * ~192 k points ~ 384000
#
# NOTE on missing values:
#   The bounding box is a rectangle aligned to the dataset extent and covers
#   significantly more area than Austria's actual land mass. Roughly half of
#   all grid points fall outside Austrian territory and are returned as null.
# =============================================================================

TARGET_DATE = "2026-04-13"
SNOWGRID_URL = "https://dataset.api.hub.geosphere.at/v1/grid/historical/snowgrid_cl-v2-1d-1km"
AUSTRIA_BBOX = "46.16,9.39,49.18,17.38"
SNOWGRID_PARAMETERS = ["snow_depth", "swe_tot"]

# cache for the query results per day
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "snowgrid")

def fetch_snowgrid(date_str: str) -> tuple[dict, bool]:
    cache_path = os.path.join(CACHE_DIR, f"{date_str}.json")

    # from cache if possible
    if os.path.isfile(cache_path):
        logger.info(f"Loading cached response from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f), True

    # NOTE: Single request to not run into rate limits
    params = {
        "parameters": SNOWGRID_PARAMETERS,
        "start": f"{date_str}T00:00",
        "end": f"{date_str}T00:00",
        "bbox": AUSTRIA_BBOX,
        "output_format": "geojson",
    }

    logger.info(f"Requesting SNOWGRID data for {date_str} from GeoSphere Austria API ...")
    response = requests.get(SNOWGRID_URL, params=params, timeout=120) # seems to be very slow...
    response.raise_for_status()

    data = response.json()

    # persists to cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    logger.info(f"Response cached to {cache_path}")

    return data, False


def print_metadata(data: dict, date_str: str, from_cache: bool) -> None:
    features = data.get("features", [])
    total = len(features)

    print(f"\n--- SNOWGRID Metadata: {date_str} {'(from cache)' if from_cache else '(fresh download)'} ---")
    print(f"Total grid points : {total:,}")

    if total == 0:
        print("No features found in response.")
        return

    # compute bounding box
    lons = [f["geometry"]["coordinates"][0] for f in features]
    lats = [f["geometry"]["coordinates"][1] for f in features]
    print(f"Bounding box (lon): {min(lons):.4f} ... {max(lons):.4f}")
    print(f"Bounding box (lat): {min(lats):.4f} ... {max(lats):.4f}")

    timestamps = data.get("timestamps", [])
    if timestamps:
        print(f"Timestamp          : {timestamps[0]}")

    # PER VARIABLE STATISTICS
    # props are structured as: properties.parameters.<var>.data[0]
    for var in SNOWGRID_PARAMETERS:
        values = []
        missing = 0
        for f in features:
            var_data = f.get("properties", {}).get("parameters", {}).get(var, {})
            raw = var_data.get("data") or [None]
            v = raw[0]
            unit = var_data.get("unit", "")
            if v is None:
                missing += 1
            else:
                values.append(v)

        print(f"\n  [{var}] ({unit})")
        print(f" - Valid values: {len(values):,}")
        print(f" - Missing: {missing:,}")
        if values:
            print(f" - Min: {min(values):.4f}")
            print(f" - Max: {max(values):.4f}")
            print(f" - Mean: {sum(values) / len(values):.4f}")

    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    try:
        data, from_cache = fetch_snowgrid(TARGET_DATE)
    except requests.HTTPError as e:
        logger.error(f"API request failed: {e}")
        sys.exit(1)
    except requests.RequestException as e:
        logger.error(f"Network error: {e}")
        sys.exit(1)

    print_metadata(data, TARGET_DATE, from_cache)
