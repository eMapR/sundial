#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --mem=257200
#SBATCH --cpus-per-task=48
#SBATCH --partition=ceoas
#SBATCH --nodelist=yatagarasu[01-13]
#SBATCH --output=/ceoas/emapr/sundial/adhoc_analysis/embeddings/embeddings_03.log
#SBATCH --error=/ceoas/emapr/sundial/adhoc_analysis/embeddings/embeddings_03.err
#SBATCH --job-name=embeddings_03

conda activate sundial
export EXPERIMENT_NAME=all224_glkn
export BASE_PATH=/home/ceoas/truongmy/emapr/sundial/adhoc_analysis/embeddings/

srun python /home/ceoas/truongmy/emapr/sundial/adhoc_analysis/embeddings/embeddings_03.py