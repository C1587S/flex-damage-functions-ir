#!/usr/bin/env python3
"""
F2 tables generation with vectorized operations.

This module generates F2 damage tables by applying regional polynomials
and global gamma coefficients to mortality data.
"""

import argparse
import yaml
import os
import duckdb
import pandas as pd
import numpy as np
import sys


# ================================================================
# Time period mapping
# ================================================================


def get_windows_map():
    """
    Create DataFrame mapping years to time periods and center years.

    Returns:
        DataFrame with year, period, and year_center columns
    """
    windows = {
        "2020_2039": (2020, 2039),
        "2040_2059": (2040, 2059),
        "2060_2079": (2060, 2079),
        "2080_2094": (2080, 2094),
        "2095_2100": (2095, 2100)
    }
    
    map_list = []
    for wname, (y1, y2) in windows.items():
        year_center = y2
        for y in range(y1, y2 + 1):
            map_list.append({
                "year": int(y),
                "period": wname,
                "year_center": int(year_center)
            })
    return pd.DataFrame(map_list)


def main():
    """
    Generate F2 damage tables for all SSPs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--ssp", required=False, default=None, help="Process only this SSP (e.g., SSP1)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    OUT_DIR = cfg["output_dir"]
    DB_PATH = os.path.join(OUT_DIR, "mortality_fe.duckdb")
    MEANTAS_PATH = os.path.join(os.getcwd(), "climate/meantas.csv")
    F2_OUT = os.path.join(OUT_DIR, "f2_tables")
    os.makedirs(F2_OUT, exist_ok=True)

    print("Starting vectorized F2 table generation")

    # ----
    poly_file = os.path.join(OUT_DIR, "regional_polynomials.csv")
    poly = pd.read_csv(poly_file)
    scale_factor = 1e5

    poly["alpha"]   *= scale_factor
    poly["beta"]    *= scale_factor
    poly["zeta"]    *= scale_factor
    poly["eta"]     *= scale_factor
    poly["sigma11"] *= (scale_factor**2)
    poly["sigma12"] *= (scale_factor**2)
    poly["sigma22"] *= (scale_factor**2)

    poly_slim = poly[["region", "alpha", "beta"]]

    # ----
    gamma_df = pd.read_csv(os.path.join(OUT_DIR, "gamma_statistics.csv"))
    gamma = float(gamma_df.loc[gamma_df["term"] == "gamma", "estimate"].values[0])
    print(f"Gamma loaded: {gamma}")

    # ----
    tas = pd.read_csv(MEANTAS_PATH)
    id_vars = ["year"]
    val_vars = [c for c in tas.columns if c != "year"]
    tas_melted = tas.melt(id_vars=id_vars, var_name="rcp", value_name="TT")

    win_map = get_windows_map()

    # ----
    con = duckdb.connect(DB_PATH, read_only=True)
    con.execute("PRAGMA threads=4;")
    con.execute(f"PRAGMA memory_limit='8GB';")

    if args.ssp:
        ssps_to_run = [args.ssp]
    else:
        ssps_to_run = con.execute("SELECT DISTINCT ssp FROM mort_raw_unionized ORDER BY ssp").df()["ssp"].tolist()

    print(f"Processing SSPs: {ssps_to_run}")

    # ================================================================
    # Process each SSP
    # ================================================================
    for ssp in ssps_to_run:
        out_csv = os.path.join(F2_OUT, f"{ssp}.csv")
        print(f"\nProcessing SSP: {ssp}")

        query = f"""
        WITH baseline AS (
            SELECT region, rcp, model, AVG(loggdppc) as base_loggdppc
            FROM mort_raw_unionized
            WHERE ssp = '{ssp}' AND year BETWEEN 2010 AND 2015
            GROUP BY region, rcp, model
        )
        SELECT 
            m.region, m.year, m.rcp, m.model, 
            m.delta_mortality, m.loggdppc,
            (m.loggdppc - b.base_loggdppc) as lgdp_diff
        FROM mort_raw_unionized m
        JOIN baseline b 
          ON m.region = b.region 
          AND m.rcp = b.rcp 
          AND m.model = b.model
        WHERE m.ssp = '{ssp}'
        """

        df = con.execute(query).df()

        if df.empty:
            print(f"No data for {ssp}, skipping")
            continue

        df_merged = df.merge(win_map, on="year", how="inner")

        grouped = df_merged.groupby(["region", "rcp", "model", "period", "year_center"], as_index=False).agg({
            "lgdp_diff": "mean",
            "delta_mortality": "mean"
        })

        grouped = grouped.merge(poly_slim, on="region", how="inner")

        grouped = grouped.merge(tas_melted, left_on=["year_center", "rcp"], right_on=["year", "rcp"], how="inner")

        ratio_y = np.exp(grouped["lgdp_diff"] * (110.0 / 80.0))

        term_temp = (grouped["alpha"] * grouped["TT"]) + (grouped["beta"] * (grouped["TT"]**2))
        grouped["flextotal"] = term_temp * (ratio_y ** gamma)

        grouped["rawtotal"] = grouped["delta_mortality"] * scale_factor

        grouped["ssp"] = ssp

        final_cols = ["region", "rcp", "ssp", "model", "period", "year_center", "TT", "flextotal", "rawtotal"]
        final_df = grouped[final_cols]

        final_df.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}: {len(final_df)} rows")

    con.close()
    print("\nF2 table generation completed successfully")

if __name__ == "__main__":
    main()