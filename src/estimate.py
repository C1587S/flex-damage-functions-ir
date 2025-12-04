"""
Global fixed effects estimation for mortality damage functions.

This module performs fixed effects regression to estimate the income elasticity
parameter gamma for mortality impacts.
"""

import duckdb
import os
import logging
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def run_estimation(cfg, parquet_files):
    """
    Run global fixed effects estimation pipeline.

    Args:
        cfg: Configuration dictionary containing output_dir and parameters
        parquet_files: List of paths to input parquet files
    """
    output_dir = cfg["output_dir"]
    db_path = os.path.join(output_dir, "mortality_fe.duckdb")

    con = duckdb.connect(db_path)
    con.execute(f"PRAGMA threads={cfg.get('n_workers', 8)};")
    con.execute(f"PRAGMA memory_limit='{cfg.get('memory_limit_gb', 64)}GB';")

    # ================================================================
    # Cumulative mode: mort_raw_unionized table
    # ================================================================
    raw_union_exists = con.execute("""
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_name = 'mort_raw_unionized'
    """).fetchone()[0]

    file_list_sql = ", ".join([f"'{p}'" for p in parquet_files])

    if not raw_union_exists:
        logger.info("Creating base table mort_raw_unionized from Parquet files")
        con.execute(f"""
            CREATE TABLE mort_raw_unionized AS
            SELECT * FROM read_parquet([{file_list_sql}])
        """)
        logger.info("Base table mort_raw_unionized created")
    else:
        logger.info("Appending new data into mort_raw_unionized")
        con.execute(f"""
            INSERT INTO mort_raw_unionized
            SELECT r.*
            FROM read_parquet([{file_list_sql}]) r
            LEFT JOIN mort_raw_unionized m
              ON  m.ssp    = r.ssp
              AND m.rcp    = r.rcp
              AND m.gcm    = r.gcm
              AND m.model  = r.model
              AND m.batch  = r.batch
              AND m.year   = r.year
              AND m.region = r.region
            WHERE m.region IS NULL
        """)
        logger.info("New rows appended to mort_raw_unionized")

    con.execute("""
        CREATE OR REPLACE VIEW raw_mort AS
        SELECT * FROM mort_raw_unionized
    """)

    # ================================================================
    # Data trimming and preparation
    # ================================================================
    logger.info("Building mort_analysis table with trimmed data")

    q05 = con.execute("""
        SELECT quantile_cont(log(delta_mortality), 0.05)
        FROM raw_mort
        WHERE delta_mortality > 0
    """).fetchone()[0]

    logger.info(f"Trim threshold (5th percentile log mortality): {q05}")

    logger.info("Computing trimming summary")

    trim_stats = con.execute(f"""
        SELECT
            COUNT(*) AS total_rows,

            SUM(CASE WHEN delta_mortality > 0 THEN 1 ELSE 0 END) AS positive_rows,

            SUM(CASE WHEN delta_mortality > 0
                     AND log(delta_mortality) < {q05}
                THEN 1 ELSE 0 END) AS trimmed_out,

            1.0 * SUM(CASE WHEN delta_mortality > 0
                            AND log(delta_mortality) < {q05}
                      THEN 1 ELSE 0 END)
                 / NULLIF(SUM(CASE WHEN delta_mortality > 0 THEN 1 ELSE 0 END), 0)
                 AS trimmed_fraction

        FROM raw_mort
    """).df()

    trim_stats.to_csv(os.path.join(output_dir, "trimming_summary.csv"), index=False)
    logger.info("Trimming summary saved to trimming_summary.csv")

    con.execute("DROP TABLE IF EXISTS mort_analysis")

    con.execute(f"""
        CREATE TABLE mort_analysis AS
        SELECT
            region,
            year,
            ssp,
            rcp,
            gcm,
            model,        
            batch,
            population,
            delta_mortality,
            delta_temp,
            lgdp_delta,
            loggdppc AS x,
            log(delta_mortality) AS y,
            CONCAT(
                CASE WHEN delta_mortality >= 0 THEN '+' ELSE '-' END,
                '-', region,
                '-', CAST(FLOOR(delta_temp / 0.5) AS INTEGER)
            ) AS grp

        FROM raw_mort
        WHERE delta_mortality > 0
          AND population IS NOT NULL
          AND loggdppc IS NOT NULL
          AND log(delta_mortality) >= {q05}
    """)

    logger.info("mort_analysis table materialized")

    # ================================================================
    # Pre-estimation diagnostics
    # ================================================================
    n_obs = con.execute("SELECT COUNT(*) FROM mort_analysis").fetchone()[0]
    logger.info(f"Total observations: {n_obs}")

    logger.info("Generating exploratory plots")

    sample_df = con.execute("""
        SELECT
            y, x, lgdp_delta, population, region, year, grp
        FROM mort_analysis
        USING SAMPLE 2%
    """).df()

    if "lgdp_delta" in sample_df:
        plt.figure(figsize=(10, 6))
        plt.scatter(sample_df["lgdp_delta"], sample_df["y"], s=5, alpha=0.3)
        plt.xlabel("log GDPpc – baseline (lgdp_delta)")
        plt.ylabel("log mortality")
        plt.title("Pre-FE: log mortality vs lgdp_delta")
        plt.grid(alpha=0.2)
        plt.savefig(os.path.join(output_dir, "plot_preFE_lgdp_vs_logmort.png"))
        plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(sample_df["y"], bins=40, edgecolor="black", alpha=0.7)
    plt.title("Histogram: log mortality (pre-FE)")
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(output_dir, "plot_hist_logmort_preFE.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(sample_df["x"], bins=40, edgecolor="black", alpha=0.7)
    plt.title("Histogram: x = loggdppc (pre-FE)")
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(output_dir, "plot_hist_regressor_preFE.png"))
    plt.close()

    diag = {
        "n_obs": n_obs,
        "n_groups": con.execute("SELECT COUNT(DISTINCT grp) FROM mort_analysis").fetchone()[0],
        "year_range": con.execute("SELECT MIN(year), MAX(year) FROM mort_analysis").fetchone(),
        "mean_population": con.execute("SELECT AVG(population) FROM mort_analysis").fetchone()[0],
        "regressor_min": con.execute("SELECT MIN(x) FROM mort_analysis").fetchone()[0],
        "regressor_max": con.execute("SELECT MAX(x) FROM mort_analysis").fetchone()[0],
    }

    with open(os.path.join(output_dir, "diagnostics_estimation.json"), "w") as f:
        json.dump(diag, f, indent=4)

    if n_obs < 100:
        logger.error("Not enough observations for fixed effects estimation. Aborting.")
        con.close()
        return

    # ================================================================
    # Global temperature-mortality aggregation
    # ================================================================
    logger.info("Computing global temperature-mortality dataset")

    global_df = con.execute("""
        WITH agg AS (
            SELECT
                year,
                rcp,
                ssp,
                gcm,
                model,
                SUM(population * delta_mortality)           AS w_mort,
                SUM(population * delta_temp)                AS w_temp,
                SUM(population * exp(lgdp_delta))           AS w_gdp,
                SUM(population)                             AS population
            FROM mort_analysis
            GROUP BY year, rcp, ssp, gcm, model
        )
        SELECT
            year,
            rcp,
            ssp,
            gcm,
            model,
            population,
            w_mort / population AS mean,
            w_temp / population AS tas_preind,
            w_gdp  / population AS gdp_diff
        FROM agg
        ORDER BY year, rcp, ssp, gcm, model
    """).df()

    global_df.to_csv(os.path.join(output_dir, "global_globaldf.csv"), index=False)

    # ================================================================
    # Fixed effects estimation and gamma computation
    # ================================================================
    logger.info("Computing fixed effects group means")

    con.execute("""
        CREATE OR REPLACE TABLE fe_grp AS
        SELECT grp,
               SUM(population * y) / SUM(population) AS y_bar,
               SUM(population * x) / SUM(population) AS x_bar
        FROM mort_analysis
        GROUP BY grp
    """)

    con.execute("""
        CREATE OR REPLACE TABLE fe_year AS
        SELECT year,
               SUM(population * y) / SUM(population) AS y_bar,
               SUM(population * x) / SUM(population) AS x_bar
        FROM mort_analysis
        GROUP BY year
    """)

    con.execute("""
        CREATE OR REPLACE TABLE fe_glob AS
        SELECT
            SUM(population * y) / SUM(population) AS y_bar,
            SUM(population * x) / SUM(population) AS x_bar
        FROM mort_analysis
    """)

    logger.info("Estimating gamma coefficient using fixed effects demeaned variables")

    num, den = con.execute("""
        WITH dm AS (
            SELECT
                m.population,
                (m.y - g.y_bar - yr.y_bar + gl.y_bar) AS y_dm,
                (m.x - g.x_bar - yr.x_bar + gl.x_bar) AS x_dm
            FROM mort_analysis m
            JOIN fe_grp  g  ON m.grp  = g.grp
            JOIN fe_year yr ON m.year = yr.year
            CROSS JOIN fe_glob gl
        )
        SELECT
            SUM(population * x_dm * y_dm),
            SUM(population * x_dm * x_dm)
        FROM dm
    """).fetchone()

    if den == 0 or den is None or np.isnan(den):
        logger.error("Denominator is zero, cannot estimate gamma coefficient")
        con.close()
        return

    gamma = num / den
    logger.info(f"Gamma coefficient: {gamma}")

    dm_sample = con.execute("""
        WITH dm AS (
            SELECT
                m.y,
                m.x,
                (m.y - g.y_bar - yr.y_bar + gl.y_bar) AS y_dm,
                (m.x - g.x_bar - yr.x_bar + gl.x_bar) AS x_dm
            FROM mort_analysis m
            JOIN fe_grp  g  ON m.grp  = g.grp
            JOIN fe_year yr ON m.year = yr.year
            CROSS JOIN fe_glob gl
        )
        SELECT * FROM dm USING SAMPLE 2%
    """).df()

    plt.figure(figsize=(10, 6))
    plt.scatter(dm_sample["x_dm"], dm_sample["y_dm"], s=5, alpha=0.3)
    plt.xlabel("x_dm")
    plt.ylabel("y_dm")
    plt.title("Post-FE scatter: x_dm vs y_dm")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "plot_scatter_dm.png"))
    plt.close()

    res = dm_sample["y_dm"] - gamma * dm_sample["x_dm"]

    plt.figure(figsize=(10, 6))
    plt.hist(res, bins=40, edgecolor="black", alpha=0.7)
    plt.title("Residual histogram")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "plot_residuals_hist.png"))
    plt.close()

    from scipy.stats import probplot
    plt.figure(figsize=(8, 8))
    probplot(res, plot=plt)
    plt.title("Residual QQ plot")
    plt.savefig(os.path.join(output_dir, "plot_residuals_qq.png"))
    plt.close()

    # ================================================================
    # Robust standard error computation
    # ================================================================
    meat = con.execute(f"""
        WITH dm AS (
            SELECT
                m.population,
                (m.y - g.y_bar - yr.y_bar + gl.y_bar) AS y_dm,
                (m.x - g.x_bar - yr.x_bar + gl.x_bar) AS x_dm
            FROM mort_analysis m
            JOIN fe_grp  g  ON m.grp  = g.grp
            JOIN fe_year yr ON m.year = yr.year
            CROSS JOIN fe_glob gl
        )
        SELECT
            SUM( POW(population * x_dm * (y_dm - {gamma} * x_dm), 2) )
        FROM dm
    """).fetchone()[0]

    gamma_se = np.sqrt(meat / (den ** 2))
    logger.info(f"Gamma standard error: {gamma_se}")

    # ================================================================
    # Global quadratic model and diagnostic plots
    # ================================================================
    global_df["mean_normed"] = global_df["mean"] / (global_df["gdp_diff"] ** gamma)
    global_df["tas_preind2"] = global_df["tas_preind"] ** 2

    X = sm.add_constant(global_df[["tas_preind", "tas_preind2"]])
    w = global_df["population"].values

    global_df = global_df.replace([np.inf, -np.inf], np.nan)
    global_df = global_df.dropna(subset=["mean_normed", "tas_preind", "tas_preind2", "population"])
    global_df = global_df[global_df["gdp_diff"] > 0]

    X = sm.add_constant(global_df[["tas_preind", "tas_preind2"]])
    w = global_df["population"].values

    model = sm.WLS(global_df["mean_normed"], X, weights=w).fit()
    global_df["resids"] = model.resid

    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=global_df,
        x="tas_preind",
        y="mean_normed",
        alpha=0.5,
        s=25,
        edgecolor=None,
    )

    xs = np.linspace(global_df["tas_preind"].min(), global_df["tas_preind"].max(), 200)
    X_pred = sm.add_constant(
        pd.DataFrame({"tas_preind": xs, "tas_preind2": xs ** 2})
    )
    ys = model.predict(X_pred)

    plt.plot(xs, ys, color="red", linewidth=2, label="Quadratic WLS fit")
    plt.xlabel("Temperature anomaly (°C)")
    plt.ylabel("Normalized mortality (mean_normed)")
    plt.title("Global mortality response to temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mortality_temp_relationship.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "mortality_temp_relationship.pdf"))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=global_df,
        x="year",
        y="resids",
        alpha=0.5,
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
    )
    plt.xlabel("Year")
    plt.ylabel("Residuals")
    plt.title("Global model residuals over time")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals_time.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "residuals_time.pdf"))
    plt.close()

    global_df.to_csv(os.path.join(output_dir, "globaldf_with_resids.csv"), index=False)

    # ================================================================
    # Output files generation
    # ================================================================
    stats_df = pd.DataFrame({
        "term": ["gamma"],
        "estimate": [gamma],
        "std_error": [gamma_se],
        "t_stat": [gamma / gamma_se],
        "n_obs": [n_obs],
    })
    stats_df.to_csv(os.path.join(output_dir, "gamma_statistics.csv"), index=False)

    quantiles = np.linspace(0.01, 0.99, 99)
    gamma_vals = norm.ppf(quantiles, loc=gamma, scale=gamma_se)
    pd.DataFrame({"quantile": quantiles, "gamma_value": gamma_vals}) \
        .to_csv(os.path.join(output_dir, "gamma_values.csv"), index=False)

    if cfg.get("generate_plots", True):
        plt.figure(figsize=(10, 6))
        plt.hist(gamma_vals, bins=30, edgecolor="black", alpha=0.7)
        plt.axvline(gamma, color="red", linestyle="--", linewidth=2)
        plt.title(f"Gamma distribution (SE={gamma_se:.4f})")
        plt.xlabel("Gamma")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, "gamma_distribution.png"), dpi=150)
        plt.close()

    con.close()
    logger.info("Estimation complete")


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--parquets", nargs="*", default=None,
                        help="Optionally pass parquet paths manually. If omitted, uses cfg['input_dir']")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.parquets:
        parquet_files = args.parquets
    else:
        input_dir = cfg.get("input_dir")
        if input_dir is None:
            raise RuntimeError("input_dir must be defined in config.yaml OR pass --parquets manually.")

        parquet_files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".parquet")
        ]

    print("\n=================================")
    print("  RUNNING ESTIMATION STANDALONE")
    print("=================================\n")
    print("Config:", args.config)
    print("N parquet files:", len(parquet_files))

    run_estimation(cfg, parquet_files)