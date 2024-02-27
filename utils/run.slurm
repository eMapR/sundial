#!/bin/bash -l

#SBATCH --export=ALL
#SBATCH --job-name=sundial.run
#SBATCH --cpus-per-task=32
#SBATCH --get-user-env

conda activate ${SUNDIAL_ENV_NAME}

if [[ -n ${SUNDIAL_CONFIG} ]]; then
    python $SUNDIAL_BASE_PATH/src/runner.py ${METHOD} --config ${SUNDIAL_CONFIG}
else
    python $SUNDIAL_BASE_PATH/src/runner.py ${METHOD}
fi