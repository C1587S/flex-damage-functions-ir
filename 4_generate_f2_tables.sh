#!/bin/bash
# Generate F2 damage tables for specific SSPs in parallel

module load python
source activate /project/cil/home_dirs/scadavidsanchez/envs/regional-scc

# Uncomment SSPs as needed for parallel execution
# python generate_f2.py --config config.yaml --ssp SSP1 &
# python generate_f2.py --config config.yaml --ssp SSP2 &
python generate_f2.py --config config.yaml --ssp SSP3 &
# python generate_f2.py --config config.yaml --ssp SSP4 &
# python generate_f2.py --config config.yaml --ssp SSP5 &

wait
echo "F2 table generation completed"
