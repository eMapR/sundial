#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --mem=257200
#SBATCH --cpus-per-task=48
#SBATCH --partition=ceoas
#SBATCH --nodelist=yatagarasu05,yatagarasu06,yatagarasu07,yatagarasu08,yatagarasu09
#SBATCH --output=/ceoas/emapr/sundial/utils/glkn/embeddings_03.log
#SBATCH --error=/ceoas/emapr/sundial/utils/glkn/embeddings_03.err
#SBATCH --job-name=embeddings_03

conda activate sundial
export EXPERIMENT_NAME=all224_glkn
export BASE_PATH=/home/ceoas/truongmy/emapr/sundial/utils/glkn/

srun python /home/ceoas/truongmy/emapr/sundial/utils/glkn/embeddings_03.py