#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --mem=257200
#SBATCH --cpus-per-task=48
#SBATCH --partition=ceoas
#SBATCH --nodelist=yatagarasu[01-12]

conda activate sundial

export BASE_PATH="/home/ceoas/truongmy/emapr/sundial/adhoc_analysis/embeddings/"
export EXPERIMENT_NAME="all224_glkn"

srun /home/ceoas/truongmy/miniconda3_x86/envs/sundial/bin/python /home/ceoas/truongmy/emapr/sundial/adhoc_analysis/embeddings/embeddings_00.py && \
srun /home/ceoas/truongmy/miniconda3_x86/envs/sundial/bin/python /home/ceoas/truongmy/emapr/sundial/adhoc_analysis/embeddings/embeddings_01.py && \
srun /home/ceoas/truongmy/miniconda3_x86/envs/sundial/bin/python /home/ceoas/truongmy/emapr/sundial/adhoc_analysis/embeddings/embeddings_02.py