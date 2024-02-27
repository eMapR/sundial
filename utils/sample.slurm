#!/bin/bash -l

#SBATCH --export=ALL
#SBATCH --job-name=sundial.sample
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --get-user-env

conda activate ${SUNDIAL_ENV_NAME}

if [[ -n ${SUNDIAL_CONFIG} ]]; then
    python $SUNDIAL_BASE_PATH/src/utils/sampler.py --config_path ${SUNDIAL_CONFIG}
else
    python $SUNDIAL_BASE_PATH/src/utils/sampler.py 
fi