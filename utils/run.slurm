#!/bin/bash -l

#SBATCH --job-name=sundial.run

conda activate ${SUNDIAL_ENV_NAME}

python ${SUNDIAL_BASE_PATH}/src/runner.py

