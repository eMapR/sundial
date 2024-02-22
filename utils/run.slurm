#!/bin/bash -l

#SBATCH --export=ALL
#SBATCH --job-name=sundial.run
#SBATCH --cpus-per-task=32
#SBATCH --get-user-env

export SUNDIAL_BASE_PATH=${SUNDIAL_BASE_PATH}
export SUNDIAL_SAMPLE_NAME=${SUNDIAL_SAMPLE_NAME}
conda activate ${SUNDIAL_ENV_NAME}

if [[ -n ${SUNDIAL_CONFIG} ]]; then
    ./run.sh -m ${METHOD} -c ${SUNDIAL_CONFIG}
else
    ./run.sh -m ${METHOD}
fi
