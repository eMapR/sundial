#!/bin/bash -l

#SBATCH --export=ALL
#SBATCH --job-name=sundial.sample
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --get-user-env

export SUNDIAL_BASE_PATH=${SUNDIAL_BASE_PATH}
export SUNDIAL_SAMPLE_NAME=${SUNDIAL_SAMPLE_NAME}
conda activate ${SUNDIAL_ENV_NAME}

if [[ -n ${SUNDIAL_CONFIG} ]]; then
    ./sample.sh -c ${SUNDIAL_CONFIG}
else
    ./sample.sh 
fi
