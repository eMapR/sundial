#!/bin/bash -l

#SBATCH --job-name=sundial.${SUNDIAL_METHOD}
#SBATCH --output=${SUNDIAL_BASE_PATH}/logs/${SUNDIAL_METHOD}.o
#SBATCH --error=${SUNDIAL_BASE_PATH}/logs/${SUNDIAL_METHOD}.e

#SBATCH --partition=${SUNDIAL_PARTITION}
#SBATCH --chdir=${SUNDIAL_BASE_PATH}
#SBATCH --export=ALL

conda activate ${SUNDIAL_ENV_NAME}

python ${SUNDIAL_BASE_PATH}/src/runner.py

