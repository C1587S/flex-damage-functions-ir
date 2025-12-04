#!/usr/bin/env python3
"""
Regional preparation validation script.

Validates that global estimation outputs exist before launching regional jobs.
"""

import os
import yaml
import sys


def main():
    """
    Validate global estimation outputs for regional analysis.
    """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    OUT = cfg["output_dir"]

    gdf = os.path.join(OUT, "globaldf_with_resids.csv")
    gammaf = os.path.join(OUT, "gamma_statistics.csv")

    if not os.path.exists(gdf):
        print("Missing globaldf_with_resids.csv. Run global estimation first.")
        sys.exit(1)

    if not os.path.exists(gammaf):
        print("Missing gamma_statistics.csv. Run global estimation first.")
        sys.exit(1)

    print("Global inputs validated successfully")
    print("Ready to launch regional array jobs")


if __name__ == "__main__":
    main()
