#!/bin/bash -l

#SBATCH --export=ALL
#SBATCH --job-name=sundial.download
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32

conda activate ${SUNDIAL_ENV_NAME}

python ${SUNDIAL_BASE_PATH}/src/pipeline/downloader.py
