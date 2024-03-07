#!/bin/bash -l
#SBATCH --mem=64G

conda activate ${SUNDIAL_ENV_NAME}
python ${SUNDIAL_BASE_PATH}/src/runner.py