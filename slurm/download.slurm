#!/bin/bash -l
#SBATCH --mem=64G
#SBATCH --cpus-per-task=42

conda activate ${SUNDIAL_ENV_NAME}
srun python ${SUNDIAL_BASE_PATH}/src/runner.py
