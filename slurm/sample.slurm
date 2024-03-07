#!/bin/bash -l
#SBATCH --mem=32G

conda activate ${SUNDIAL_ENV_NAME}
python ${SUNDIAL_BASE_PATH}/src/runner.py