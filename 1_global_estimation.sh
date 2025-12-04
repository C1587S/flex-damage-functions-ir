#!/bin/bash
# Run global fixed effects estimation

module load python
source activate /project/cil/home_dirs/scadavidsanchez/envs/regional-scc-df-fit
cd /project/cil/home_dirs/scadavidsanchez/projects/flex-damage-functions-ir

python 1_global_estimation.py --config config.yaml
