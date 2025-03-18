#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1

conda activate ${SUNDIAL_ENV_NAME}
srun python ${SUNDIAL_BASE_PATH}/src/runner.py
