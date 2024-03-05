#!/bin/bash -l

#SBATCH --job-name=sundial.download
#SBATCH --mem=64G

conda activate ${SUNDIAL_ENV_NAME}

srun python ${SUNDIAL_BASE_PATH}/src/pipeline/downloader.py

