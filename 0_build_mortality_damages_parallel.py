#!/usr/bin/env python3
"""
Build integrated mortality datasets from simulation outputs using parallel processing.

This script processes mortality simulation data and aligns it with climate and
socioeconomic datasets. The approach enforces strict coordinate alignment through
reindexing to prevent artificial coordinate combinations and memory issues.

Process:
1. Mortality data defines the coordinate hypercube.
2. Climate and socioeconomic data are aligned to match mortality coordinates.
3. Data is merged with coordinate override to prevent outer join behavior.
"""

import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ====================================================================
# Configuration
# ====================================================================

CLIMATE_CSV = "/project/cil/home_dirs/scadavidsanchez/flexdamages-ir/gcm_temp_gmst_year_preind1986-2005.csv"
SOCIO_ZARR  = "/project/cil/gcp/integration_replication/inputs/econ/raw/integration-econ-bc39.zarr"

IAM_MAPPING = {"IIASA GDP": "low", "OECD Env-Growth": "high"}


# ====================================================================
# Core Processing Functions
# ====================================================================

def safe_open(path):
    """
    Open a dataset without dask chunks to avoid file handle locks.

    Ensures the file handle is closed immediately after reading by using
    the load() method within a context manager.
    """
    with xr.open_dataset(path, chunks=None) as ds:
        ds.load()
        return ds

def process_single_sim(args):
    """
    Process a single simulation directory and extract adjusted mortality.

    Loads mortality, cost, and historical climate data from a simulation
    directory, calculates adjusted mortality values, and returns a properly
    structured DataArray with all relevant coordinates.
    """
    sim_dir, batch, rcp, gcm, model, ssp = args

    try:
        agg_path  = os.path.join(sim_dir, "Agespec_interaction_response-combined.nc4")
        cost_path = os.path.join(sim_dir, "Agespec_interaction_response-combined-costs.nc4")
        hist_path = os.path.join(sim_dir, "Agespec_interaction_response-combined-histclim.nc4")

        # Verify all required files exist
        if not (os.path.exists(agg_path) and os.path.exists(cost_path) and os.path.exists(hist_path)):
            return None

        agg  = safe_open(agg_path)
        cost = safe_open(cost_path)
        hist = safe_open(hist_path)

        # Find common years across all three datasets
        years = np.intersect1d(
            agg["year"].values,
            np.intersect1d(cost["year"].values, hist["year"].values)
        )

        if len(years) == 0:
            return None

        # Subset to common years
        agg  = agg.sel(year=years)
        cost = cost.sel(year=years)
        hist = hist.sel(year=years)

        # Calculate adjusted mortality from aggregated response and cost data
        region_ids = agg["regions"].values.astype(str)
        cost_mid = (cost["costs_lb"] + cost["costs_ub"]) / 2

        mort_values = (
            agg["rebased"].values +
            (cost_mid.values / 100000.0) -
            hist["rebased"].values
        )

        # Create DataArray with year and region dimensions
        da = xr.DataArray(
            mort_values,
            dims=("year", "region"),
            coords={
                "year": years.astype(int),
                "region": region_ids
            },
            name="adjusted_mortality"
        )

        # Add scalar coordinates for batch and climate metadata
        da = da.assign_coords(
            batch=str(batch),
            rcp=str(rcp),
            gcm=str(gcm),
            model=str(model),
            ssp=str(ssp)
        )

        # Expand dimensions to create full hypercube structure
        da = da.expand_dims(["batch", "rcp", "gcm", "model", "ssp"])

        return da

    except Exception as e:
        return None

# ====================================================================
# File Discovery
# ====================================================================

def discover_sims(basepath, ssp, mode, target_batch="batch13", max_gcms=2):
    """
    Discover available simulation directories in the file structure.

    Scans the base path for batch directories, RCPs, GCMs, and SSPs,
    returning a list of task tuples for parallel processing.
    """
    tasks = []

    batches = [d for d in os.listdir(basepath) if d.startswith("batch")]
    if mode == "test":
        batches = [target_batch] if target_batch in batches else []

    print(f"Scanning batches: {batches}")

    for batch in batches:
        batch_path = os.path.join(basepath, batch)
        if not os.path.isdir(batch_path): continue

        for rcp in ["rcp45", "rcp85"]:
            rcp_path = os.path.join(batch_path, rcp)
            if not os.path.isdir(rcp_path): continue

            all_gcms = sorted([d for d in os.listdir(rcp_path) if os.path.isdir(os.path.join(rcp_path, d))])
            
            if mode == "test":
                selected_gcms = all_gcms[:max_gcms]
            else:
                selected_gcms = all_gcms

            for gcm in selected_gcms:
                gcm_path = os.path.join(rcp_path, gcm)
                for model in ["low", "high"]:
                    sim_dir = os.path.join(gcm_path, model, ssp)
                    if os.path.isdir(sim_dir):
                        tasks.append((sim_dir, batch, rcp, gcm, model, ssp))

    return tasks

# ====================================================================
# Main Builder
# ====================================================================

def build_dataset(args):
    """
    Main pipeline to build an integrated mortality dataset.

    Orchestrates: discovery, parallel loading, coordinate alignment,
    and writing to Zarr format.
    """
    basepath = args.basepath
    ssp = args.ssp
    mode = args.mode
    batch = args.batch
    output_dir = args.output_dir

    num_workers = args.cores if args.cores else os.cpu_count()
    print(f"\nRunning with {num_workers} cores")

    # ----
    # Discover simulation directories
    # ----
    print(f"\nDiscovering simulations ({mode} mode)")
    tasks = discover_sims(basepath, ssp, mode, target_batch=batch, max_gcms=2 if mode=='test' else 999)

    if not tasks:
        raise ValueError("No simulations found.")

    # ----
    # Load mortality data in parallel
    # ----
    print(f"Loading mortality data in parallel")
    mortality_das = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_sim, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(tasks), desc="Processing simulations"):
            result = f.result()
            if result is not None:
                mortality_das.append(result)

    if not mortality_das:
        raise RuntimeError("All simulations failed to load.")

    # ----
    # Combine mortality DataArrays
    # ----
    print("\nCombining DataArrays into master hypercube")
    ds_mort = xr.combine_by_coords(mortality_das)

    if ds_mort.year.dtype != int:
        ds_mort["year"] = ds_mort.year.astype(int)

    # Clean encoding to prevent unnecessary metadata
    for var in ds_mort.variables:
        ds_mort[var].encoding = {}

    print(f"Master topology established: {ds_mort.sizes}")

    # ----
    # Load and align climate data
    # ----
    print("Loading and aligning climate data")
    df_clim = pd.read_csv(CLIMATE_CSV)
    if "scenario" in df_clim.columns:
        df_clim = df_clim.rename(columns={"scenario": "rcp"})

    ds_clim = df_clim.set_index(["gcm", "rcp", "year"])["anomaly"].to_xarray()
    ds_clim.name = "temperature_anomaly"
    ds_clim["year"] = ds_clim.year.astype(int)

    # Align climate coordinates to match mortality dataset
    target_clim_coords = {
        "gcm": ds_mort.coords["gcm"],
        "rcp": ds_mort.coords["rcp"],
        "year": ds_mort.coords["year"]
    }

    ds_clim_aligned = ds_clim.reindex(target_clim_coords)

    # ----
    # Load and align socioeconomic data
    # ----
    print("Loading and aligning socioeconomic data")
    ds_socio = xr.open_zarr(SOCIO_ZARR)

    # Remove unnecessary variables
    if "gdp" in ds_socio:
        ds_socio = ds_socio.drop_vars("gdp")
    if ds_socio.year.dtype != int:
        ds_socio["year"] = ds_socio.year.astype(int)

    # Map IAM model names to standard labels
    new_labels = [IAM_MAPPING.get(str(m), str(m)) for m in ds_socio.model.values]
    ds_socio = ds_socio.assign_coords(model=new_labels)

    # Align socioeconomic coordinates to match mortality dataset
    target_socio_coords = {
        "ssp": ds_mort.coords["ssp"],
        "region": ds_mort.coords["region"],
        "model": ds_mort.coords["model"],
        "year": ds_mort.coords["year"]
    }

    ds_socio = ds_socio.sel(ssp=ssp)
    ds_socio_aligned = ds_socio.reindex(target_socio_coords)

    # ----
    # Merge datasets
    # ----
    print("Merging datasets with coordinate override")

    ds_final = xr.merge(
        [ds_mort, ds_clim_aligned, ds_socio_aligned],
        join="override",
        compat="override"
    )

    # ----
    # Optimize dimension ordering and chunking
    # ----
    desired = ["batch", "ssp", "rcp", "gcm", "model", "year", "region"]
    existing_dims = [d for d in desired if d in ds_final.dims]
    ds_final = ds_final.transpose(*existing_dims)

    # Filter to year 2010 and later
    ds_final = ds_final.sel(year=ds_final.year >= 2010)

    # Apply chunking strategy: region is large, year is kept as single chunk for efficiency
    regions = ds_final.region.values
    chunk_r = min(2000, len(regions))

    chunks = {
        d: 1 for d in existing_dims if d not in ["year", "region"]
    }
    chunks["year"] = -1
    chunks["region"] = chunk_r

    ds_final = ds_final.chunk(chunks)

    # ----
    # Write to Zarr
    # ----
    os.makedirs(output_dir, exist_ok=True)
    outname = f"mortality_{ssp}_TEST.zarr" if mode == "test" else f"mortality_{ssp}.zarr"
    outpath = os.path.join(output_dir, outname)

    print(f"Writing dataset to {outpath}")
    ds_final.to_zarr(outpath, mode="w", consolidated=True)
    print("Dataset written successfully.")

    return outpath

# ====================================================================
# Validation
# ====================================================================

def validate_output(zarr_path):
    """
    Validate the output Zarr dataset for structure and completeness.

    Checks dimensions, coordinate structure, and samples data to ensure
    proper alignment and completeness.
    """
    print("\nValidating output dataset\n")
    ds = xr.open_zarr(zarr_path)
    print(ds)

    print("\nChecking dimensions and coordinates:")
    print(f"Coordinates: {list(ds.coords)}")

    # ----
    # Sample data verification
    # ----
    print("\nSampling data from ABW region, year 2099:")
    try:
        check = ds.sel(region="ABW", year=2099)

        print("Temperature anomaly:")
        print(check['temperature_anomaly'].values)

        if "gdp" in check:
            print("ERROR: GDP variable should have been removed.")
        else:
            print("Verification passed: GDP removed successfully.")

    except Exception as e:
        print(f"Sample verification failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Mortality Zarr")
    parser.add_argument("--basepath", required=True)
    parser.add_argument("--ssp", required=True)
    parser.add_argument("--mode", choices=["full", "test"], required=True)
    parser.add_argument("--batch", default="batch13")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cores", type=int, default=None, help="Num parallel workers")

    args = parser.parse_args()

    outpath = build_dataset(args)

    validate_output(outpath)