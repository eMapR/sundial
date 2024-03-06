#!/bin/bash -l

#SBATCH --job-name=sundial.run
#SBATCH --gres=gpu:gh200_480gb:1

conda activate ${SUNDIAL_ENV_NAME}

python ${SUNDIAL_BASE_PATH}/src/runner.py

