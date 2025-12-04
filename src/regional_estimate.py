#!/usr/bin/env python3
"""
Regional estimation module with parallel processing optimization.

This module provides optimized regional polynomial estimation using:
- Batch SQL loading for multiple regions
- Parallel processing with joblib
- Vectorized calculations
- Efficient residual merging
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List
import duckdb
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ================================================================
# Processing functions
# ================================================================


def fit_polynomial(T: np.ndarray, y: np.ndarray, lam: float = 1e-8) -> Dict[str, Any]:
    """
    Fit polynomial with ridge regularization and convexity enforcement.

    Args:
        T: Temperature anomaly array
        y: Normalized mortality array
        lam: Ridge regularization parameter

    Returns:
        Dictionary with alpha, beta, sigma parameters, residuals, and sample size
    """
    n = len(y)
    T2 = T ** 2

    Z = np.column_stack([T, T2])
    XtX = Z.T @ Z + lam * np.eye(2)
    Xty = Z.T @ y

    try:
        coeff = np.linalg.solve(XtX, Xty)
        alpha, beta = float(coeff[0]), float(coeff[1])
    except np.linalg.LinAlgError:
        denom = (T @ T) + lam
        if denom == 0:
            return None
        alpha = float((T @ y) / denom)
        beta = 0.0

    if beta < 0:
        denom = (T @ T) + lam
        if denom == 0:
            return None
        alpha = float((T @ y) / denom)
        beta = 0.0
        
        fit = alpha * T
        resid_local = y - fit
        p_lin = 1
        
        if n > p_lin:
            s2 = float((resid_local @ resid_local) / (n - p_lin))
        else:
            s2 = 0.0
        
        XtX_lin = np.array([[(T @ T) + lam]])
        try:
            inv_XtX_lin = np.linalg.inv(XtX_lin)
        except np.linalg.LinAlgError:
            inv_XtX_lin = np.array([[np.nan]])
        
        vcv = np.zeros((2, 2))
        vcv[0, 0] = s2 * inv_XtX_lin[0, 0]
    else:
        fit = alpha * T + beta * T2
        resid_local = y - fit
        p = 2
        
        if n > p:
            s2 = float((resid_local @ resid_local) / (n - p))
        else:
            s2 = 0.0
        
        try:
            inv_XtX = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            inv_XtX = np.full((2, 2), np.nan)
        
        vcv = s2 * inv_XtX
    
    return {
        "alpha": alpha,
        "beta": beta,
        "sigma11": float(vcv[0, 0]),
        "sigma12": float(vcv[0, 1]),
        "sigma22": float(vcv[1, 1]),
        "resid_local": resid_local,
        "n": n,
    }


def compute_rho(resid_local: np.ndarray, resid_global: np.ndarray) -> float:
    """
    Compute correlation between local and global residuals.

    Args:
        resid_local: Regional residuals
        resid_global: Global residuals

    Returns:
        Correlation coefficient
    """
    mask = np.isfinite(resid_local) & np.isfinite(resid_global)
    if mask.sum() < 3:
        return np.nan
    
    rl = resid_local[mask]
    rg = resid_global[mask]
    
    if np.std(rl) == 0 or np.std(rg) == 0:
        return np.nan
    
    return float(np.corrcoef(rl, rg)[0, 1])


def compute_heteroskedasticity(resid_local: np.ndarray, T: np.ndarray, lam: float = 1e-8) -> tuple:
    """
    Compute heteroskedasticity parameters.

    Args:
        resid_local: Regional residuals
        T: Temperature anomaly array
        lam: Ridge regularization parameter

    Returns:
        Tuple of (zeta, eta) parameters
    """
    totalsd = np.abs(resid_local)
    mask = np.isfinite(totalsd) & np.isfinite(T)
    
    if mask.sum() < 2:
        return np.nan, np.nan
    
    T_sd = T[mask]
    s_sd = totalsd[mask]
    
    denom = (T_sd @ T_sd) + lam
    if denom == 0:
        return np.nan, np.nan
    
    zeta = float((T_sd @ s_sd) / denom)
    resid_sd = s_sd - zeta * T_sd
    
    if len(resid_sd) > 1:
        eta = float(np.std(resid_sd, ddof=1))
    else:
        eta = np.nan
    
    return zeta, eta


def process_single_region(
    region: str,
    df_region: pd.DataFrame,
    gamma_mu: float,
    global_index: Optional[pd.Series],
    test_mode: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Process a single region and return results dictionary.

    Args:
        region: Region identifier
        df_region: DataFrame with regional data
        gamma_mu: Global gamma coefficient
        global_index: Index of global residuals
        test_mode: Whether to use test mode

    Returns:
        Dictionary with regional parameters or None if insufficient data
    """
    if df_region.empty:
        return None
    
    df = df_region.copy()

    if global_index is not None:
        merge_keys = ["year", "rcp", "ssp", "gcm", "model"]
        keys = list(zip(*[df[k].values for k in merge_keys]))
        df["resid_global"] = global_index.reindex(keys).values
    else:
        df["resid_global"] = np.nan

    if test_mode:
        df["resid_global"] = np.random.normal(0, 0.1, size=len(df))

    df = df.replace([np.inf, -np.inf], np.nan)

    with np.errstate(over="ignore", invalid="ignore"):
        df["mean_normed"] = df["delta_mortality"] / np.exp(df["lgdp_delta"] * gamma_mu)

    mask = df["mean_normed"].notna() & df["delta_temp"].notna()
    if mask.sum() < 5:
        return None

    df_fit = df.loc[mask]
    T = df_fit["delta_temp"].values.astype(float)
    y = df_fit["mean_normed"].values.astype(float)
    resid_global = df_fit["resid_global"].values.astype(float)

    fit_result = fit_polynomial(T, y)
    if fit_result is None:
        return None

    rho = compute_rho(fit_result["resid_local"], resid_global)

    zeta, eta = compute_heteroskedasticity(fit_result["resid_local"], T)
    
    return {
        "region": region,
        "alpha": fit_result["alpha"],
        "beta": fit_result["beta"],
        "sigma11": fit_result["sigma11"],
        "sigma12": fit_result["sigma12"],
        "sigma22": fit_result["sigma22"],
        "rho": rho,
        "zeta": zeta,
        "eta": eta,
        "n": fit_result["n"],
    }


# ================================================================
# Main function
# ================================================================


def main():
    """
    Run optimized regional estimation with parallel processing.
    """
    if len(sys.argv) < 3:
        print("Usage: python regional_estimate_optimized.py <config.yaml> <job_id>")
        sys.exit(1)

    config_path = sys.argv[1]
    job_id = int(sys.argv[2])

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]
    regional_outdir = os.path.join(output_dir, "regional_results")
    os.makedirs(regional_outdir, exist_ok=True)

    test_mode = cfg.get("regional_test_mode", False)
    test_region_limit = cfg.get("regional_test_regions", 50)
    regions_per_job = cfg.get("regions_per_job", 25)
    n_jobs = cfg.get("n_parallel_jobs", -1)

    logger.info(f"Regional Estimation Job {job_id} (Optimized)")
    logger.info(f"Test mode: {test_mode}")
    logger.info(f"Regions per job: {regions_per_job}")
    logger.info(f"Parallel jobs: {n_jobs}")

    # ----
    gamma_stats_path = os.path.join(output_dir, "gamma_statistics.csv")
    if not os.path.exists(gamma_stats_path):
        raise RuntimeError(f"gamma_statistics.csv not found in {output_dir}")

    gamma_stats = pd.read_csv(gamma_stats_path)

    if "mu" in gamma_stats.columns:
        gamma_mu = float(gamma_stats["mu"].iloc[0])
    else:
        row_gamma = gamma_stats.loc[gamma_stats["term"] == "gamma"]
        if row_gamma.empty:
            raise RuntimeError("gamma_statistics.csv does not contain 'mu' or term == 'gamma'.")
        gamma_mu = float(row_gamma["estimate"].iloc[0])

    logger.info(f"Using gamma coefficient: {gamma_mu:.6f}")

    # ----
    global_df_path = os.path.join(output_dir, "globaldf_with_resids.csv")
    if not os.path.exists(global_df_path):
        logger.warning("globaldf_with_resids.csv missing, regional rho will be NaN")
        global_index = None
    else:
        global_df = pd.read_csv(global_df_path)
        merge_keys = ["year", "rcp", "ssp", "gcm", "model"]
        merge_keys = [k for k in merge_keys if k in global_df.columns]

        if "resids" not in global_df.columns:
            raise RuntimeError("globaldf_with_resids.csv must contain 'resids' column")

        global_index = global_df.set_index(merge_keys)["resids"]
        logger.info(f"Global residuals loaded with merge keys: {merge_keys}")

    # ----
    db_path = os.path.join(output_dir, "mortality_fe.duckdb")
    if not os.path.exists(db_path):
        raise RuntimeError(f"mortality_fe.duckdb not found at {db_path}")

    con = duckdb.connect(db_path, read_only=True)
    con.execute("PRAGMA threads=8;")

    regions = con.execute("""
        SELECT DISTINCT region
        FROM mort_analysis
        ORDER BY region
    """).df()["region"].tolist()

    if test_mode:
        regions = regions[:test_region_limit]

    total_regions = len(regions)
    logger.info(f"Total distinct regions: {total_regions}")

    # ----
    start = job_id * regions_per_job
    end = min(start + regions_per_job, total_regions)

    if start >= total_regions:
        logger.info(f"Job {job_id}: start index {start} >= total {total_regions}. Nothing to do.")
        con.close()
        return

    batch_regions = regions[start:end]
    logger.info(f"Processing regions index {start} to {end-1} (count={len(batch_regions)})")

    if not test_mode:
        regions_to_process = []
        for r in batch_regions:
            out_file = os.path.join(regional_outdir, f"{r}.parquet")
            if not os.path.exists(out_file):
                regions_to_process.append(r)
        
        skipped = len(batch_regions) - len(regions_to_process)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already-computed regions.")
        batch_regions = regions_to_process

    if not batch_regions:
        logger.info("All regions in this batch already computed. Exiting.")
        con.close()
        return

    # ================================================================
    # Load batch regions in single query
    # ================================================================
    logger.info(f"Loading data for {len(batch_regions)} regions in single query")

    placeholders = ", ".join(["?"] * len(batch_regions))

    df_all = con.execute(f"""
        SELECT
            region, year, rcp, ssp, gcm, model,
            delta_mortality, delta_temp, lgdp_delta
        FROM mort_analysis
        WHERE region IN ({placeholders})
    """, batch_regions).df()

    con.close()

    logger.info(f"Loaded {len(df_all):,} total rows for {df_all['region'].nunique()} regions")

    # ================================================================
    # Process regions in parallel
    # ================================================================
    logger.info(f"Processing regions in parallel (n_jobs={n_jobs})")

    grouped = {region: group for region, group in df_all.groupby("region")}
    
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_region)(
            region=region,
            df_region=grouped[region],
            gamma_mu=gamma_mu,
            global_index=global_index,
            test_mode=test_mode,
        )
        for region in batch_regions
        if region in grouped
    )

    results = [r for r in results if r is not None]

    logger.info(f"Successfully processed {len(results)} regions")

    # ================================================================
    # Write results
    # ================================================================
    for result in results:
        region = result["region"]
        out_file = os.path.join(regional_outdir, f"{region}.parquet")
        pd.DataFrame([result]).to_parquet(out_file)

    logger.info("Regional Estimation Job Complete")


if __name__ == "__main__":
    main()