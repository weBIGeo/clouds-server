import os
import numpy as np
import xarray as xr
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import config to know max levels
from dwd_connect import VAR_SPECS


@dataclass
class DataCube:
    """
    Represents the loaded 3D atmosphere.
    data: Dict[var_name] -> 3D Array (Levels, Lat, Lon)
    """

    data: Dict[str, np.ndarray]
    dims: Tuple[int, int, int]  # (Levels, Lat, Lon)
    meta: Dict


def get_roi_indices(
    ds: xr.Dataset, lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> Tuple[slice, slice, Dict]:
    """
    Determines the integer array indices (slices) for the region of interest
    based on a reference dataset.
    """
    # 1. Identify Coordinate Names (Robustness against 'lat' vs 'latitude')
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    lats = ds[lat_name].values
    lons = ds[lon_name].values

    # 2. Check Coordinate Dimensionality
    # ICON-D2 Regular Grid usually loads as 1D coords in xarray/cfgrib.
    if lats.ndim != 1 or lons.ndim != 1:
        # Fallback for 2D coords (rare for 'regular-lat-lon' but possible)
        # We simplify: calculate mask on the 2D grid, find bounding box of True values
        mask = (
            (lats >= lat_min)
            & (lats <= lat_max)
            & (lons >= lon_min)
            & (lons <= lon_max)
        )
        if not np.any(mask):
            raise ValueError("Region of Interest is outside the dataset coverage.")

        # Find integer bounds of the mask
        rows, cols = np.where(mask)
        lat_slice = slice(rows.min(), rows.max() + 1)
        lon_slice = slice(cols.min(), cols.max() + 1)
    else:
        # Standard 1D Coords logic
        # Find indices where lat/lon are within bounds
        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        lon_mask = (lons >= lon_min) & (lons <= lon_max)

        if not np.any(lat_mask) or not np.any(lon_mask):
            raise ValueError("Region of Interest is outside the dataset coverage.")

        # Convert boolean mask to integer indices
        lat_indices = np.where(lat_mask)[0]
        lon_indices = np.where(lon_mask)[0]

        # Create slices (assuming contiguous region)
        lat_slice = slice(lat_indices.min(), lat_indices.max() + 1)
        lon_slice = slice(lon_indices.min(), lon_indices.max() + 1)

    # 3. Metadata for the cropped region
    # We grab the actual coordinate values of the crop
    cropped_lats = lats[lat_slice] if lats.ndim == 1 else lats[lat_slice, lon_slice]
    cropped_lons = lons[lon_slice] if lons.ndim == 1 else lons[lat_slice, lon_slice]

    meta = {
        "lat_min": float(np.min(cropped_lats)),
        "lat_max": float(np.max(cropped_lats)),
        "lon_min": float(np.min(cropped_lons)),
        "lon_max": float(np.max(cropped_lons)),
        "lat_coords": cropped_lats,
        "lon_coords": cropped_lons,
    }

    return lat_slice, lon_slice, meta


def _read_worker(path, lat_slice, lon_slice):
    """
    Reads a GRIB file in a separate process.
    Returns the numpy array crop or None on failure.
    """
    try:
        # 'indexpath': '' prevents writing .idx files to disk, saving IO time
        with xr.open_dataset(
            path, engine="cfgrib", backend_kwargs={"indexpath": ""}
        ) as ds:
            # Dynamically find the variable name (e.g., 'clc', 't', etc.)
            var_name = list(ds.data_vars)[0]
            data_var = ds[var_name]

            # Perform the crop immediately
            # .values forces the read and decode within this process
            if data_var.ndim == 2:
                return data_var.isel(latitude=lat_slice, longitude=lon_slice).values
            elif data_var.ndim == 3:
                return data_var.isel(
                    step=0, latitude=lat_slice, longitude=lon_slice
                ).values
    except Exception as e:
        return None


# --- 2. Updated load_region function ---
def load_region(
    download_results: Dict[str, List[str]],
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> DataCube:
    print("--- Loading and Stacking GRIBs ---")

    # 1. Reference Logic
    ref_path = next((paths[0] for k, paths in download_results.items() if paths), None)
    if not ref_path:
        raise FileNotFoundError("No downloaded files found.")

    with xr.open_dataset(
        ref_path, engine="cfgrib", backend_kwargs={"indexpath": ""}
    ) as ds:
        lat_slice, lon_slice, meta = get_roi_indices(
            ds, lat_min, lat_max, lon_min, lon_max
        )

    ny = lat_slice.stop - lat_slice.start
    nx = lon_slice.stop - lon_slice.start
    print(f"    Grid Crop: {nx}x{ny} pixels")

    stacked_data = {}

    # 2. Parallel Loading
    # Adjust lower if you run out of RAM.
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:

        for var_key, file_paths in download_results.items():
            if not file_paths:
                continue

            config = VAR_SPECS[var_key]

            # Map levels
            if config.has_levels:
                nz = len(config.levels)
                level_to_idx = {lvl: i for i, lvl in enumerate(config.levels)}
            else:
                nz = 1
                level_to_idx = {0: 0}

            # Pre-allocate array
            arr_3d = np.zeros((nz, ny, nx), dtype=np.float32)

            # Submit Tasks
            # We map {Future -> z_index} to know where to put the result
            future_to_idx = {}

            for path in file_paths:
                # Parse level from filename
                # (Assuming standard format tile_processor logic produces)
                try:
                    parts = os.path.basename(path).split(".")[0].split("_")
                    level_val = int(parts[-1])
                except ValueError:
                    continue

                if level_val in level_to_idx:
                    z_idx = level_to_idx[level_val]
                    # Submit to process pool
                    future = executor.submit(_read_worker, path, lat_slice, lon_slice)
                    future_to_idx[future] = z_idx

            print(f"    Processing {var_key.upper()} ({len(future_to_idx)} tasks)...")

            # Collect Results
            for future in as_completed(future_to_idx):
                z_idx = future_to_idx[future]
                data = future.result()

                if data is not None:
                    # Write into the pre-allocated array
                    arr_3d[z_idx, :, :] = data

            stacked_data[var_key] = arr_3d

    print("    Stacking complete.")
    return DataCube(stacked_data, (65, ny, nx), meta)
