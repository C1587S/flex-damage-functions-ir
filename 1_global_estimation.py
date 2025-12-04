#!/usr/bin/env python3
"""
Mortality estimation pipeline orchestrator.

Manages the complete workflow dor gamma estimation: data extraction from Zarr, conversion to Parquet,
statistical estimation using DuckDB and fixed effects models, and gamma distribution
fitting.
"""
import argparse
import yaml
import os
import sys
import logging
import json
from src import extract, estimate

# ====================================================================
# Logging Configuration
# ====================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_config(path):
    """
    Load configuration from a YAML file.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """
    Execute the complete mortality estimation pipeline.

    Stages:
    1. Extract data from Zarr sources and write to Parquet
    2. Run statistical estimation with DuckDB and generate diagnostics
    """
    parser = argparse.ArgumentParser(description="Mortality FE Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg['output_dir'], exist_ok=True)

    # ----
    # Save configuration for reproducibility
    # ----
    with open(os.path.join(cfg['output_dir'], "run_metadata.json"), "w") as f:
        json.dump(cfg, f, indent=4)

    logger.info("Starting mortality estimation pipeline")

    # ----
    # Stage 1: Extract and prepare data
    # ----
    logger.info("Running data extraction")
    parquet_files = extract.run_extraction(cfg)

    if not parquet_files:
        logger.error("No parquet files generated. Pipeline halted.")
        sys.exit(1)

    # ----
    # Stage 2: Run statistical estimation
    # ----
    logger.info("Running statistical estimation")
    estimate.run_estimation(cfg, parquet_files)

    logger.info("Pipeline complete")

if __name__ == "__main__":
    main()