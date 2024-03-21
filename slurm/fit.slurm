#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --mem=0
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1

##export NCCL_P2P_DISABLE=1
##export NCCL_DEBUG=INFO

conda activate ${SUNDIAL_ENV_NAME}
srun python ${SUNDIAL_BASE_PATH}/src/runner.py
