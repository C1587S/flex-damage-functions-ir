#!/usr/bin/env python3
"""
Regional polynomial aggregation with parallel processing.

This module aggregates regional estimation results from thousands of parquet files
using parallel processing and efficient memory management.

Usage:
    python regional_aggregate.py --config config.yaml
    python regional_aggregate.py --config config.yaml --workers 16 --batch-size 1000
    python regional_aggregate.py --config config.yaml --skip-plots
"""

import argparse
import yaml
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import List, Tuple
import warnings

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

# Plotting
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para paralelización
import matplotlib.pyplot as plt
import seaborn as sns

# Progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Tip: install tqdm for progress bars: pip install tqdm")


# ================================================================
# Configuration
# ================================================================
MAX_WORKERS = None
BATCH_SIZE = 500

PARAMS = ["alpha", "beta", "rho", "sigma11", "sigma12",
          "sigma22", "zeta", "eta"]


# ================================================================
# Plotting configuration
# ================================================================


def set_ggplot_style():
    sns.set_theme(context="talk")
    plt.rcParams.update({
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.color": "grey",
        "grid.alpha": 0.3,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.titleweight": "bold",
        "axes.titlepad": 14,
        "axes.labelpad": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ================================================================
# Parallel file reading functions
# ================================================================


def read_single_parquet(filepath: str) -> pa.Table:
    """
    Read a single parquet file and return PyArrow table.

    Args:
        filepath: Path to parquet file

    Returns:
        PyArrow table or None if error
    """
    try:
        return pq.read_table(filepath)
    except Exception as e:
        warnings.warn(f"Error reading {filepath}: {e}")
        return None


def read_parquet_batch(filepaths: List[str], n_workers: int) -> pa.Table:
    """
    Read a batch of parquet files in parallel.

    Args:
        filepaths: List of file paths to read
        n_workers: Number of parallel workers

    Returns:
        Concatenated PyArrow table
    """
    tables = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(read_single_parquet, fp): fp 
                   for fp in filepaths}
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                tables.append(result)
    
    if not tables:
        return None
    
    return pa.concat_tables(tables)


def read_all_parquets_batched(
    filepaths: List[str],
    batch_size: int = BATCH_SIZE,
    n_workers: int = None
) -> pd.DataFrame:
    """
    Read all parquet files in parallel batches.

    Args:
        filepaths: List of file paths
        batch_size: Number of files per batch
        n_workers: Number of parallel workers

    Returns:
        Combined DataFrame
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 4
    
    n_files = len(filepaths)
    n_batches = (n_files + batch_size - 1) // batch_size

    print(f"Reading {n_files:,} files in {n_batches} batches")
    print(f"Workers: {n_workers}, Batch size: {batch_size}")

    all_tables = []

    iterator = range(0, n_files, batch_size)
    if HAS_TQDM:
        iterator = tqdm(iterator, desc="Batches", total=n_batches)

    for i in iterator:
        batch_files = filepaths[i:i + batch_size]
        table = read_parquet_batch(batch_files, n_workers)
        if table is not None:
            all_tables.append(table)

    print("Concatenating tables")
    combined = pa.concat_tables(all_tables)

    print("Converting to DataFrame")
    return combined.to_pandas()


# ================================================================
# Parallel plotting functions
# ================================================================


def save_single_histogram(args: Tuple[str, np.ndarray, str]) -> str:
    """
    Generate and save a single histogram.

    Args:
        args: Tuple of (parameter name, data array, output directory)

    Returns:
        Parameter name
    """
    param, data, plot_dir = args

    set_ggplot_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(
        data,
        bins=50,
        color="#4682B4",
        edgecolor="black",
        alpha=0.8
    )

    ax.set_title(f"Distribution of {param}", weight="bold")
    ax.set_xlabel(param)
    ax.set_ylabel("Count")

    fig.tight_layout()

    out_path = os.path.join(plot_dir, f"dist_{param}.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return param


def generate_histograms_parallel(
    df: pd.DataFrame,
    params: List[str],
    plot_dir: str,
    n_workers: int = None
) -> None:
    """
    Generate histograms in parallel.

    Args:
        df: DataFrame with parameters
        params: List of parameter names
        plot_dir: Output directory for plots
        n_workers: Number of parallel workers
    """
    if n_workers is None:
        n_workers = min(len(params), os.cpu_count() or 4)
    
    plot_args = [
        (param, df[param].dropna().values, plot_dir)
        for param in params
        if param in df.columns
    ]

    print(f"Generating {len(plot_args)} histograms with {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        if HAS_TQDM:
            list(tqdm(
                executor.map(save_single_histogram, plot_args),
                total=len(plot_args),
                desc="Histograms"
            ))
        else:
            list(executor.map(save_single_histogram, plot_args))


def generate_facet_plot(
    df: pd.DataFrame,
    params: List[str],
    plot_dir: str
) -> None:
    """
    Generate faceted parameter distribution plot.

    Args:
        df: DataFrame with parameters
        params: List of parameter names
        plot_dir: Output directory for plots
    """
    set_ggplot_style()

    valid_params = [p for p in params if p in df.columns]

    melted = df.melt(
        value_vars=valid_params,
        var_name="parameter",
        value_name="value"
    )

    if len(melted) > 1_000_000:
        print(f"Subsampling facet plot: {len(melted):,} to 1,000,000 points")
        melted = melted.sample(n=1_000_000, random_state=42)
    
    g = sns.FacetGrid(
        melted,
        col="parameter",
        col_wrap=4,
        sharex=False,
        sharey=False,
        height=3.8
    )
    
    g.map_dataframe(
        sns.histplot,
        x="value",
        bins=40,
        color="#4682B4",
        edgecolor="black",
        alpha=0.8
    )
    
    for ax in g.axes.flatten():
        ax.grid(True, alpha=0.25)
        ax.set_facecolor("white")
    
    plt.subplots_adjust(top=0.88)
    g.fig.suptitle("Regional Parameter Distributions", weight="bold", fontsize=18)
    
    facet_path = os.path.join(plot_dir, "facet_parameter_distributions.png")
    plt.savefig(facet_path, dpi=160)
    plt.close()
    
    print(f"Saved: {facet_path}")


# ================================================================
# Main function
# ================================================================


def main():
    """
    Main aggregation function for regional polynomials.
    """
    parser = argparse.ArgumentParser(
        description="Regional aggregation with parallel processing"
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of workers (default: all cores)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Files per batch (default: {BATCH_SIZE})")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    n_workers = args.workers or os.cpu_count() or 4
    batch_size = args.batch_size

    print(f"\n{'='*60}")
    print(f"REGIONAL AGGREGATION (OPTIMIZED)")
    print(f"{'='*60}")
    print(f"Workers: {n_workers}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}\n")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]
    test_mode = cfg.get("regional_test_mode", False)
    regional_dir = os.path.join(output_dir, "regional_results")

    if not os.path.exists(regional_dir):
        raise RuntimeError(f"Directory does not exist: {regional_dir}")

    files = sorted(
        f for f in os.listdir(regional_dir)
        if f.endswith(".parquet")
    )

    if len(files) == 0:
        print("No parquet files found")
        return

    print(f"Found: {len(files):,} parquet files\n")
    
    filepaths = [os.path.join(regional_dir, f) for f in files]

    # ----
    full = read_all_parquets_batched(
        filepaths,
        batch_size=batch_size,
        n_workers=n_workers
    )

    print(f"\nFinal DataFrame: {len(full):,} rows, {len(full.columns)} columns")
    print(f"Memory: {full.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")

    # ----
    out_csv = os.path.join(output_dir, "regional_polynomials.csv")
    print(f"Saving CSV: {out_csv}")
    full.to_csv(out_csv, index=False)

    # ----
    summary = {
        "n_regions": int(full["region"].nunique()) if "region" in full.columns else len(full),
        "n_rows": len(full),
        "n_files": len(files),
    }
    
    for param in PARAMS:
        if param in full.columns:
            summary[f"{param}_mean"] = float(full[param].mean())
            summary[f"{param}_std"] = float(full[param].std())

    summary_path = os.path.join(output_dir, "regional_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved: {summary_path}")

    # ----
    if not args.skip_plots:
        plot_dir = os.path.join(output_dir, "regional_plots")
        os.makedirs(plot_dir, exist_ok=True)

        generate_histograms_parallel(full, PARAMS, plot_dir, n_workers)
        print("Individual histograms completed")

        print("Generating facet plot")
        generate_facet_plot(full, PARAMS, plot_dir)
    else:
        print("Plots skipped (--skip-plots)")

    if test_mode:
        print("\n=== TEST MODE ===")
        print(full.head(12))
        print("\nStatistics:")
        print(full[PARAMS].describe().T)

    print(f"\n{'='*60}")
    print("AGGREGATION COMPLETED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()