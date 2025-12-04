#!/usr/bin/env python3
"""
Comprehensive diagnostics for damage function estimation.

This module generates diagnostic plots and summaries for:
- Gamma coefficient distribution
- Global mortality-temperature model
- Regional polynomial parameters
- F2 table outputs

Usage:
    python src/alphafit_diagnostics.py --config config.yaml

Requirements:
- gamma_statistics.csv
- gamma_values.csv
- global_globaldf.csv (and optionally globaldf_with_resids.csv)
- regional_polynomials.csv
- f2_tables/*.csv
"""

import argparse
import os
import json
import glob

import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import normaltest, probplot


# ================================================================
# Plotting configuration
# ================================================================


def set_ggplot_style():
    """
    Configure plotting style similar to ggplot2.
    """
    sns.set_theme(
        style="whitegrid",
        context="talk",
        palette="deep"
    )
    plt.rcParams.update({
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 120,
    })


# ================================================================
# Utility functions
# ================================================================


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def safe_to_numeric(df, cols):
    """
    Convert columns to numeric with error coercion.

    Args:
        df: DataFrame
        cols: List of column names

    Returns:
        DataFrame with converted columns
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ================================================================
# Gamma diagnostics
# ================================================================


def gamma_diagnostics(output_dir, diag_dir):
    """Generate gamma coefficient diagnostics."""
    print("\nGenerating gamma diagnostics")

    gamma_csv = os.path.join(output_dir, "gamma_statistics.csv")
    values_csv = os.path.join(output_dir, "gamma_values.csv")

    if not os.path.exists(gamma_csv) or not os.path.exists(values_csv):
        print("Gamma files missing, skipping gamma diagnostics")
        return

    gamma_df = pd.read_csv(gamma_csv)
    vals_df = pd.read_csv(values_csv)

    gamma_row = gamma_df.loc[gamma_df["term"] == "gamma"].iloc[0]
    gamma_hat = float(gamma_row["estimate"])
    gamma_se = float(gamma_row["std_error"])
    gamma_values = vals_df["gamma_value"].dropna().values

    if len(gamma_values) >= 8:
        stat, pval = normaltest(gamma_values)
    else:
        stat, pval = np.nan, np.nan

    summary = {
        "gamma_hat": gamma_hat,
        "gamma_se": gamma_se,
        "normaltest_stat": float(stat),
        "normaltest_pvalue": float(pval),
        "n_gamma_values": int(len(gamma_values)),
    }

    with open(os.path.join(diag_dir, "gamma_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    set_ggplot_style()
    plt.figure(figsize=(8, 6))
    sns.histplot(gamma_values, bins=30, kde=True)
    plt.axvline(gamma_hat, color="red", linestyle="--", linewidth=2, label="gamma_hat")
    plt.title("Gamma value distribution")
    plt.xlabel("gamma_value")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, "gamma_distribution_diag.png"))
    plt.close()

    plt.figure(figsize=(6, 6))
    probplot(gamma_values, dist="norm", plot=plt)
    plt.title("Gamma values QQ-plot")
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, "gamma_qqplot.png"))
    plt.close()

    print("Gamma diagnostics saved")


# ================================================================
# Global model diagnostics
# ================================================================


def global_diagnostics(output_dir, diag_dir):
    """Generate global model diagnostics."""
    print("\nGenerating global model diagnostics")

    global_path = os.path.join(output_dir, "global_globaldf.csv")
    global_resid_path = os.path.join(output_dir, "globaldf_with_resids.csv")
    gamma_csv = os.path.join(output_dir, "gamma_statistics.csv")

    if not os.path.exists(global_path) or not os.path.exists(gamma_csv):
        print("Global files missing, skipping global diagnostics")
        return

    gamma_df = pd.read_csv(gamma_csv)
    gamma_hat = float(gamma_df.loc[gamma_df["term"] == "gamma", "estimate"].iloc[0])

    if os.path.exists(global_resid_path):
        global_df = pd.read_csv(global_resid_path)
        has_resids = "resids" in global_df.columns
    else:
        global_df = pd.read_csv(global_path)
        has_resids = False

    needed = ["tas_preind", "mean", "population", "gdp_diff", "year"]
    if not all(c in global_df.columns for c in needed):
        print("Missing required columns in global dataframe")
        return

    if "mean_normed" not in global_df.columns or not has_resids:
        global_df["mean_normed"] = global_df["mean"] / (global_df["gdp_diff"] ** gamma_hat)

        import statsmodels.api as sm

        global_df["tas_preind2"] = global_df["tas_preind"] ** 2
        X = sm.add_constant(global_df[["tas_preind", "tas_preind2"]])
        w = global_df["population"].values
        model = sm.WLS(global_df["mean_normed"], X, weights=w).fit()
        global_df["resids"] = model.resid

        global_df.to_csv(global_resid_path, index=False)

    set_ggplot_style()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=global_df,
        x="tas_preind",
        y="mean_normed",
        alpha=0.4,
        s=25,
        edgecolor=None,
    )
    sns.regplot(
        data=global_df,
        x="tas_preind",
        y="mean_normed",
        scatter=False,
        order=2,
        color="red",
        line_kws={"linewidth": 2},
    )
    plt.xlabel("Temperature anomaly (°C)")
    plt.ylabel("Normalized mortality (mean_normed)")
    plt.title("Global mortality response to temperature")
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, "global_mortality_temp_relationship.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=global_df,
        x="year",
        y="resids",
        alpha=0.4,
        s=25,
        edgecolor=None,
    )
    sns.regplot(
        data=global_df,
        x="year",
        y="resids",
        scatter=False,
        lowess=True,
        color="red",
        line_kws={"linewidth": 2},
    )
    plt.xlabel("Year")
    plt.ylabel("Residuals")
    plt.title("Global model residuals over time")
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, "global_residuals_time.png"))
    plt.close()

    resid_summary = {
        "resid_mean": float(global_df["resids"].mean()),
        "resid_std": float(global_df["resids"].std()),
        "resid_min": float(global_df["resids"].min()),
        "resid_max": float(global_df["resids"].max()),
        "n": int(global_df["resids"].notna().sum()),
    }
    with open(os.path.join(diag_dir, "global_residuals_summary.json"), "w") as f:
        json.dump(resid_summary, f, indent=4)

    print("Global diagnostics saved")


# ================================================================
# Regional polynomial diagnostics
# ================================================================


def regional_polynomial_diagnostics(output_dir, diag_dir):
    """Generate regional polynomial parameter diagnostics."""
    print("\nGenerating regional polynomial diagnostics")

    poly_path = os.path.join(output_dir, "regional_polynomials.csv")
    if not os.path.exists(poly_path):
        print("Regional polynomials file missing, skipping regional diagnostics")
        return

    poly = pd.read_csv(poly_path)
    num_cols = ["alpha", "beta", "rho", "sigma11", "sigma12", "sigma22", "zeta", "eta"]
    poly = safe_to_numeric(poly, num_cols)

    desc = poly[num_cols].describe().to_dict()
    with open(os.path.join(diag_dir, "regional_parameters_summary.json"), "w") as f:
        json.dump(desc, f, indent=4, default=float)

    set_ggplot_style()
    for param in num_cols:
        if param not in poly.columns:
            continue
        plt.figure(figsize=(8, 5))
        sns.histplot(poly[param].dropna(), bins=40, kde=True)
        plt.xlabel(param)
        plt.ylabel("Count")
        plt.title(f"Distribution of {param}")
        plt.tight_layout()
        plt.savefig(os.path.join(diag_dir, f"dist_{param}.png"))
        plt.close()

    melted = poly.melt(value_vars=num_cols, var_name="parameter", value_name="value").dropna()
    plt.figure(figsize=(14, 10))
    g = sns.FacetGrid(melted, col="parameter", col_wrap=4, sharex=False, sharey=False)
    g.map_dataframe(sns.histplot, x="value", bins=40)
    g.set_titles("{col_name}")
    g.fig.suptitle("Regional parameter distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, "facet_parameter_distributions.png"))
    plt.close()

    poly_x = poly.copy()
    poly_x["T_cross"] = np.where(
        poly_x["beta"] != 0,
        -poly_x["alpha"] / poly_x["beta"],
        np.nan
    )

    def categorize_row(row):
        b = row["beta"]
        T = row["T_cross"]
        if pd.isna(b) or b == 0:
            return "beta_zero"
        if pd.isna(T):
            return "no_cross"
        if T < 0:
            return "cross_below_0"
        if T > 20:
            return "cross_above_20"
        return "cross_0_20"

    poly_x["cross_category"] = poly_x.apply(categorize_row, axis=1)

    counts = poly_x["cross_category"].value_counts(dropna=False)
    total = counts.sum()
    frac = (counts / total).to_dict()

    cross_summary = {
        "counts": {k: int(v) for k, v in counts.to_dict().items()},
        "fractions": {k: float(v) for k, v in frac.items()},
        "total_regions": int(total),
    }

    with open(os.path.join(diag_dir, "regional_crossing_summary.json"), "w") as f:
        json.dump(cross_summary, f, indent=4)

    # Histograma de T_cross (solo donde haya)
    cross_valid = poly_x.loc[poly_x["T_cross"].notna() & np.isfinite(poly_x["T_cross"])]
    if len(cross_valid) > 0:
        plt.figure(figsize=(8, 5))
        sns.histplot(cross_valid["T_cross"], bins=40, kde=True)
        plt.axvspan(0, 20, color="orange", alpha=0.1, label="[0,20] °C")
        plt.xlabel("T_cross = -alpha / beta (°C)")
        plt.ylabel("Count")
        plt.title("Distribution of crossing temperatures")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(diag_dir, "T_cross_distribution.png"))
        plt.close()

    norm_results = {}
    for p in ["alpha", "beta", "zeta", "eta"]:
        vals = poly[p].dropna().values
        if len(vals) >= 8:
            stat, pval = normaltest(vals)
            norm_results[p] = {"stat": float(stat), "pvalue": float(pval), "n": int(len(vals))}
        else:
            norm_results[p] = {"stat": None, "pvalue": None, "n": int(len(vals))}

    with open(os.path.join(diag_dir, "regional_normality_tests.json"), "w") as f:
        json.dump(norm_results, f, indent=4)

    print("Regional polynomial diagnostics saved")


# ================================================================
# F2 table diagnostics
# ================================================================


def f2_diagnostics(output_dir, diag_dir):
    """Generate F2 table diagnostics."""
    print("\nGenerating F2 diagnostics")

    f2_dir = os.path.join(output_dir, "f2_tables")
    if not os.path.exists(f2_dir):
        print("F2 tables directory missing, skipping F2 diagnostics")
        return

    files = sorted(glob.glob(os.path.join(f2_dir, "*.csv")))
    if not files:
        print("No F2 CSV files found, skipping F2 diagnostics")
        return

    f2_list = []
    for fpath in files:
        df = pd.read_csv(fpath)
        df["__source_file"] = os.path.basename(fpath)
        f2_list.append(df)
    f2 = pd.concat(f2_list, ignore_index=True)

    f2 = safe_to_numeric(f2, ["flextotal", "rawtotal", "TT"])
    f2 = f2.dropna(subset=["flextotal", "rawtotal"])

    f2["diff"] = f2["flextotal"] - f2["rawtotal"]

    diff_summary = {
        "mean_diff": float(f2["diff"].mean()),
        "std_diff": float(f2["diff"].std()),
        "min_diff": float(f2["diff"].min()),
        "max_diff": float(f2["diff"].max()),
        "n": int(f2["diff"].notna().sum()),
    }
    with open(os.path.join(diag_dir, "f2_diff_summary.json"), "w") as f:
        json.dump(diff_summary, f, indent=4)

    if len(f2["diff"]) >= 8:
        stat, pval = normaltest(f2["diff"].values)
        diff_norm = {"stat": float(stat), "pvalue": float(pval), "n": int(len(f2["diff"]))}
    else:
        diff_norm = {"stat": None, "pvalue": None, "n": int(len(f2["diff"]))}

    with open(os.path.join(diag_dir, "f2_diff_normality.json"), "w") as f:
        json.dump(diff_norm, f, indent=4)

    set_ggplot_style()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=f2.sample(min(len(f2), 2000), random_state=0),
        x="rawtotal",
        y="flextotal",
        alpha=0.5,
        s=20,
        edgecolor=None,
    )
    lim_min = np.nanmin([f2["rawtotal"].min(), f2["flextotal"].min()])
    lim_max = np.nanmax([f2["rawtotal"].max(), f2["flextotal"].max()])
    plt.plot([lim_min, lim_max], [lim_min, lim_max], color="red", linestyle="--", linewidth=1.5, label="1:1 line")
    plt.xlabel("Raw total (per 100k)")
    plt.ylabel("Flexible total (per 100k)")
    plt.title("F2: Flexible vs Raw totals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, "f2_scatter_flex_vs_raw.png"))
    plt.close()

    # 4.2 Histograma de diff
    plt.figure(figsize=(8, 6))
    sns.histplot(f2["diff"], bins=50, kde=True)
    plt.xlabel("flextotal - rawtotal")
    plt.ylabel("Count")
    plt.title("F2: Difference (flex - raw)")
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, "f2_diff_hist.png"))
    plt.close()

    # 4.3 QQ-plot de diff
    plt.figure(figsize=(6, 6))
    probplot(f2["diff"].values, dist="norm", plot=plt)
    plt.title("F2: diff QQ-plot")
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, "f2_diff_qqplot.png"))
    plt.close()

    if "ssp" in f2.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=f2, x="ssp", y="diff")
        plt.xlabel("SSP")
        plt.ylabel("flextotal - rawtotal")
        plt.title("F2: Error distribution by SSP")
        plt.tight_layout()
        plt.savefig(os.path.join(diag_dir, "f2_diff_by_ssp.png"))
        plt.close()

    print("F2 diagnostics saved")


# ================================================================
# Main function
# ================================================================


def main():
    """
    Run all diagnostics for damage function estimation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.yaml"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output_dir"]
    diag_dir = os.path.join(output_dir, "alphafit_diagnostics")
    ensure_dir(diag_dir)

    print("\n==========================")
    print("   ALPHAFIT DIAGNOSTICS")
    print("==========================")
    print(f"Output directory: {output_dir}")
    print(f"Diagnostics directory: {diag_dir}")

    gamma_diagnostics(output_dir, diag_dir)
    global_diagnostics(output_dir, diag_dir)
    regional_polynomial_diagnostics(output_dir, diag_dir)
    f2_diagnostics(output_dir, diag_dir)

    print("\nAll diagnostics completed")


if __name__ == "__main__":
    main()
