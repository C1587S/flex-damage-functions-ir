#!/bin/bash
# Run global fixed effects estimation

module load python
source activate /project/cil/home_dirs/scadavidsanchez/envs/regional-scc-df-fit

python 1_global_estimation.py --config config.yaml
