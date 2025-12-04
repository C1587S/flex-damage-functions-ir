#!/bin/bash
# Validate global estimation outputs before launching regional jobs

module load python
source activate /project/cil/home_dirs/scadavidsanchez/envs/regional-scc-df-fit

python src/regional_prepare.py --config config.yaml
