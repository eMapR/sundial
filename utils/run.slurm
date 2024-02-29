#!/bin/bash -l

#SBATCH --export=ALL
#SBATCH --job-name=sundial.run
#SBATCH --cpus-per-task=32

conda activate ${SUNDIAL_ENV_NAME}

python ${SUNDIAL_BASE_PATH}/src/runner.py ${SUNDIAL_METHOD} --config ${SUNDIAL_CONFIG} --ckpt_path ${SUNDIAL_CKPT_PATH}