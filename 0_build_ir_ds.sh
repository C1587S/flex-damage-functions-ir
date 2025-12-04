#!/bin/bash
# Build mortality damage dataset from Zarr archives for all SSPs

module load python
source activate /project/cil/home_dirs/scadavidsanchez/envs/regional-scc
# SSP3 SSP2 SSP4 SSP5
for SSP in  SSP1
do
    echo "Processing SSP: $SSP"

    python 0_build_mortality_damages_parallel.py \
        --mode full \
        --cores 30 \
        --ssp $SSP \
        --basepath /project/cil/battuta-shares-S3-archive/gcp/outputs/mortality/impacts-darwin/montecarlo \
        --output-dir /project/cil/home_dirs/scadavidsanchez/projects/flex-damage-functions-ir/dataset/mortality/damages

    echo "Completed SSP: $SSP"
    echo "--------------------------------------"
done
