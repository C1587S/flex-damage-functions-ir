#!/bin/bash
#SBATCH --job-name=regional_fast
#SBATCH --account=cil
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --array=0-499
#SBATCH -o slurm_out/regfast_%A_%a.out

# SLURM array job for regional estimation
# Each task processes a batch of regions

module load python
source activate /project/cil/home_dirs/scadavidsanchez/envs/regional-scc-df-fit

python src/regional_estimate.py config.yaml $SLURM_ARRAY_TASK_ID

# Test mode for first batch: python src/regional_estimate_fast.py config.yaml 0
