#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1

conda activate ${SUNDIAL_ENV_NAME}
srun python ${SUNDIAL_BASE_PATH}/src/runner.py