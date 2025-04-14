#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --mem=257200
#SBATCH --cpus-per-task=48
#SBATCH --partition=ceoas
##SBATCH --nodelist=yatagarasu[01-12]
#SBATCH --output=/ceoas/emapr/sundial/adhoc_analysis/embeddings/logs/embeddings_05.log
#SBATCH --error=/ceoas/emapr/sundial/adhoc_analysis/embeddings/logs/embeddings_05.err
#SBATCH --job-name=embeddings_05

conda activate sundial

export BASE_PATH=/home/ceoas/truongmy/emapr/sundial/adhoc_analysis/reverse_time_embeddings_expt/

srun python /home/ceoas/truongmy/emapr/sundial/adhoc_analysis/reverse_time_embeddings_expt/embeddings_05.py