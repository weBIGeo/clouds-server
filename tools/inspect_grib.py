import sys
import os
import eccodes
import xarray as xr


def inspect_raw_eccodes(filepath):
    """
    Iterates through every message in the GRIB file and prints metadata.
    Also counts messages, layers, and variables.
    """
    print(f"\n--- 🔍 RAW ECCODES INVENTORY: {os.path.basename(filepath)} ---")

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    variable_count = {}
    total_messages = 0

    with open(filepath, "rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break  # End of file

            total_messages += 1
            short_name = eccodes.codes_get(gid, "shortName")
            level_type = eccodes.codes_get(gid, "typeOfLevel")
            level_val = eccodes.codes_get(gid, "level")

            # Count occurrences per variable
            variable_count.setdefault(short_name, []).append(level_val)

            print(
                f"\n[Message {total_messages}] {short_name} @ {level_type} ({level_val})"
            )
            keys_to_check = [
                "shortName",
                "name",
                "typeOfLevel",
                "level",
                "dataDate",
                "dataTime",
                "stepRange",
                "gridType",
                "Ni",
                "Nj",
            ]

            for key in keys_to_check:
                try:
                    val = eccodes.codes_get(gid, key)
                    print(f"  {key:<15}: {val}")
                except eccodes.CodesInternalError:
                    print(f"  {key:<15}: (Not found)")

            values = eccodes.codes_get_values(gid)
            val_min = values.min()
            val_max = values.max()
            print(f"  Value Range    : {val_min:.4f} to {val_max:.4f}")

            eccodes.codes_release(gid)

    print("\n--- SUMMARY ---")
    print(f"Total messages: {total_messages}")
    print(f"Variables found: {list(variable_count.keys())}")
    for var, levels in variable_count.items():
        print(f"  {var:<10}: {len(levels)} layer(s) -> Levels: {levels}")


def inspect_xarray(filepath):
    """
    Uses xarray + cfgrib to summarize the dataset.
    """
    print(f"\n--- 🐍 XARRAY DATASET STRUCTURE ---")

    try:
        ds = xr.open_dataset(filepath, engine="cfgrib")
        print("✅ Opened successfully with default settings.")
        print(ds)

        data_vars = list(ds.data_vars)
        coords = list(ds.coords)
        print(f"\nCoords: {coords}")
        print(f"Data Vars: {data_vars}")
        print("\nVariable summary:")
        for var in data_vars:
            print(f"  {var:<15}: shape={ds[var].shape}, dims={ds[var].dims}")

    except Exception as e:
        print(f"⚠️ Default open failed: {e}")
        print("   Trying typeOfLevel filters...")

        filters = [
            "surface",
            "heightAboveGround",
            "atmosphere",
            "isobaricInhPa",
            "nominalTop",
        ]
        for level_type in filters:
            try:
                ds = xr.open_dataset(
                    filepath,
                    engine="cfgrib",
                    backend_kwargs={"filter_by_keys": {"typeOfLevel": level_type}},
                )
                data_vars = list(ds.data_vars)
                print(f"\nFilter: typeOfLevel={level_type}")
                print(f"  Data Vars: {data_vars}")
                for var in data_vars:
                    print(f"    {var:<15}: shape={ds[var].shape}, dims={ds[var].dims}")
            except Exception:
                pass


if __name__ == "__main__":
    target = sys.argv[1]

    inspect_raw_eccodes(target)
    inspect_xarray(target)
