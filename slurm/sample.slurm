#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

conda activate ${SUNDIAL_ENV_NAME}
srun python ${SUNDIAL_BASE_PATH}/src/runner.py
