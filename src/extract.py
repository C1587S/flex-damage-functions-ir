"""
Data extraction module for mortality impacts from Zarr archives.

This module extracts mortality, climate, and economic data from Zarr files
and prepares them for fixed effects estimation.
"""

import xarray as xr
import pandas as pd
import numpy as np
import logging
import os
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_zarr_path(base_dir, ssp):
    """
    Locate the Zarr file for a specific SSP.

    Args:
        base_dir: Base directory containing Zarr files
        ssp: SSP scenario identifier

    Returns:
        Path to the Zarr file
    """
    path = os.path.join(base_dir, f"mortality_{ssp}.zarr")
    if not os.path.exists(path):
        path = os.path.join(base_dir, ssp, "mortality.zarr")
    return path


def run_extraction(cfg):
    """
    Extract and prepare mortality data from Zarr archives.

    Args:
        cfg: Configuration dictionary containing extraction parameters

    Returns:
        List of extracted parquet file paths
    """

    output_dir = cfg["output_dir"]
    test_mode = cfg.get("test_mode", False)
    target_ssps = cfg["ssps"]
    target_rcps = cfg["rcps"]

    os.makedirs(output_dir, exist_ok=True)

    extracted_files = []
    global_diag = {
        "total_rows_written": 0,
        "per_batch_counts": {},
        "per_ssp_counts": {},
        "dropped_rows_merge_failure": 0,
        "dropped_rows_nan": 0,
        "dropped_rows_invalid_gdp": 0,
    }

    for ssp in target_ssps:
        zarr_path = get_zarr_path(cfg["zarr_base_dir"], ssp)
        if not os.path.exists(zarr_path):
            logger.warning(f"Zarr file not found for SSP {ssp}: {zarr_path}")
            continue

        logger.info(f"Opening Zarr file for SSP {ssp}: {zarr_path}")
        try:
            ds = xr.open_zarr(zarr_path, chunks=None)
        except Exception as e:
            logger.error(f"Cannot open Zarr file for SSP {ssp}: {e}")
            continue

        # ----
        logger.info(f"Building climate table for SSP {ssp}")
        clim = ds[["temperature_anomaly"]].sel(rcp=target_rcps)
        if test_mode:
            all_gcms = clim.gcm.values
            keep_gcms = all_gcms[: cfg.get("test_gcm_count", 2)]
            clim = clim.sel(gcm=keep_gcms)
        else:
            keep_gcms = clim.gcm.values

        df_clim = clim.to_dataframe().reset_index()
        df_clim.rename(columns={"temperature_anomaly": "delta_temp"}, inplace=True)
        df_clim["year"] = df_clim["year"].astype(int)

        # ----
        econ_vars = ["gdppc", "pop"]
        if "population" in ds:
            econ_vars = ["gdppc", "population"]

        logger.info(f"Loading economic variables for SSP {ssp}")
        econ_ds = ds[econ_vars].load()

        if "gdppc" in econ_ds:
            econ_ds["gdppc"] = econ_ds["gdppc"].where(econ_ds["gdppc"] > 0)
        else:
            raise RuntimeError("GDP variable missing in dataset")

        econ_ds["loggdppc"] = np.log(econ_ds["gdppc"])

        logger.info(f"Computing GDP baseline for SSP {ssp} (2010-2015)")
        base_slice = econ_ds["loggdppc"].sel(year=slice(2010, 2015))
        gdp_baseline = base_slice.mean(dim="year")
        gdp_baseline.name = "avg_gdp_2010_2015"

        econ_ds = xr.merge([econ_ds, gdp_baseline])

        df_econ = econ_ds.to_dataframe().reset_index()
        df_econ["year"] = df_econ["year"].astype(int)
        if "pop" in df_econ.columns:
            df_econ.rename(columns={"pop": "population"}, inplace=True)

        # ----
        batches = ds.batch.values
        if test_mode:
            batches = [cfg.get("test_batch", "batch0")]

        global_diag["per_ssp_counts"][ssp] = 0

        for batch in tqdm(batches, desc=f"Extracting {ssp}"):
            out_path = os.path.join(output_dir, f"mortality_{ssp}_{batch}.parquet")

            if os.path.exists(out_path):
                extracted_files.append(out_path)
                logger.info(f"Skipping existing file: {out_path}")
                continue

            per_batch_rows = 0
            dfs = []

            for gcm in keep_gcms:
                try:
                    raw_slice = ds[["adjusted_mortality"]].sel(
                        ssp=ssp,
                        batch=batch,
                        rcp=target_rcps,
                        gcm=gcm,
                    )
                    df = raw_slice.to_dataframe().reset_index()
                    df.rename(columns={"adjusted_mortality": "delta_mortality"}, inplace=True)
                    df["year"] = df["year"].astype(int)

                    before = len(df)
                    df = df.merge(df_econ,
                                  on=["model", "year", "region"],
                                  how="inner")
                    lost = before - len(df)
                    global_diag["dropped_rows_merge_failure"] += lost

                    before = len(df)
                    df = df.merge(df_clim,
                                  on=["rcp", "gcm", "year"],
                                  how="inner")
                    lost = before - len(df)
                    global_diag["dropped_rows_merge_failure"] += lost

                    df["lgdp_delta"] = df["loggdppc"] - df["avg_gdp_2010_2015"]

                    nan_before = len(df)
                    df = df.dropna(subset=["delta_mortality", "population", "loggdppc"])
                    nan_after = len(df)
                    global_diag["dropped_rows_nan"] += nan_before - nan_after

                    dfs.append(df)
                    per_batch_rows += len(df)

                except Exception as e:
                    logger.warning(f"Error processing batch {batch}, GCM {gcm}: {e}")
                    continue

            if dfs:
                final_df = pd.concat(dfs, ignore_index=True)
                final_df.to_parquet(out_path, index=False, compression="snappy")

                extracted_files.append(out_path)
                global_diag["per_batch_counts"][out_path] = per_batch_rows
                global_diag["total_rows_written"] += per_batch_rows
                global_diag["per_ssp_counts"][ssp] += per_batch_rows

    diag_file = os.path.join(output_dir, "diagnostics_extraction.json")
    with open(diag_file, "w") as f:
        json.dump(global_diag, f, indent=4)

    logger.info(f"Diagnostics saved to {diag_file}")

    return extracted_files
