#!/bin/bash
# Aggregate regional estimation results into single CSV

module load python
source activate /project/cil/home_dirs/scadavidsanchez/envs/regional-scc

python src/generate_f2.py --config config.yaml
