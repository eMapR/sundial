#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --cpus-per-task=64

conda activate ${SUNDIAL_ENV_NAME}
srun python ${SUNDIAL_BASE_PATH}/src/runner.py
