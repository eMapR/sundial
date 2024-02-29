#!/bin/bash -l

#SBATCH --export=ALL
#SBATCH --job-name=sundial.sample
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

conda activate ${SUNDIAL_ENV_NAME}

python ${SUNDIAL_BASE_PATH}/src/pipeline/sample.py --config $(SUNDIAL_CONFIG)