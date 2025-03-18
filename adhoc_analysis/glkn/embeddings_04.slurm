#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --partition=ceoas-arm
#SBATCH --nodelist=youmu
#SBATCH --output=/ceoas/emapr/sundial/utils/glkn/embeddings_04.log
#SBATCH --error=/ceoas/emapr/sundial/utils/glkn/embeddings_04.err
#SBATCH --job-name=embeddings_04

conda activate sundial
export EXPERIMENT_NAME="all224_glkn"
export BASE_PATH="/home/ceoas/truongmy/emapr/sundial/utils/glkn/"

srun python /home/ceoas/truongmy/emapr/sundial/utils/glkn/embeddings_04.py