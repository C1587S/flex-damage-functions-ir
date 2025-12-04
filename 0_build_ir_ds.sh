#!/bin/bash
# Build mortality damage dataset from Zarr archives for all SSPs

module load python
source activate /project/cil/home_dirs/scadavidsanchez/envs/regional-scc

for SSP in SSP3 SSP2 SSP4 SSP5 SSP1
do
    echo "Processing SSP: $SSP"

    python 0_build_mortality_damages_parallel.py \
        --mode full \
        --cores 30 \
        --ssp $SSP \
        --basepath /project/cil/battuta-shares-S3-archive/gcp/outputs/mortality/impacts-darwin/montecarlo \
        --output-dir /project/cil/home_dirs/scadavidsanchez/flexible-damage-funcs/dataset/mortality/damages

    echo "Completed SSP: $SSP"
    echo "--------------------------------------"
done
