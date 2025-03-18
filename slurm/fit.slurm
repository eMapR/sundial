#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1

## Depending on infrastructure setup, you may need to uncomment the following lines for DDP
##export NCCL_P2P_DISABLE=1
##export NCCL_DEBUG=INFO

conda activate ${SUNDIAL_ENV_NAME}
srun python ${SUNDIAL_BASE_PATH}/src/runner.py
