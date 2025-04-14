#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --mem=257200
#SBATCH --cpus-per-task=48
#SBATCH --output=/ceoas/emapr/sundial/adhoc_analysis/embeddings/logs/embeddings_04.log
#SBATCH --error=/ceoas/emapr/sundial/adhoc_analysis/embeddings/logs/embeddings_04.err
#SBATCH --partition=ceoas
#SBATCH --nodelist=yatagarasu[01-12]

conda activate sundial
export EXPERIMENT_NAME="all224_glkn"
export BASE_PATH="/home/ceoas/truongmy/emapr/sundial/adhoc_analysis/reverse_time_embeddings_expt/"

srun python /home/ceoas/truongmy/emapr/sundial/adhoc_analysis/reverse_time_embeddings_expt/embeddings_04.py