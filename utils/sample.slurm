#!/bin/bash -l

#SBATCH --job-name=sundial.sample
#SBATCH --mem=32G

conda activate ${SUNDIAL_ENV_NAME}

srun python ${SUNDIAL_BASE_PATH}/src/pipeline/sampler.py

