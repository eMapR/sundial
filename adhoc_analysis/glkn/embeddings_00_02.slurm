#!/bin/bash -l
#SBATCH --job-name=unet3dmid_dice_embed
#SBATCH --output=/home/ceoas/truongmy/emapr/sundial/utils/glkn/unet3dmid_dice_embed.log
#SBATCH --error=/home/ceoas/truongmy/emapr/sundial/utils/glkn/unet3dmid_dice_embed.err
#SBATCH --nodes=1
#SBATCH --mem=257200
#SBATCH --cpus-per-task=48
#SBATCH --partition=ceoas
#SBATCH --nodelist=yatagarasu05,yatagarasu06,yatagarasu07,yatagarasu08,yatagarasu09,yatagarasu10,yatagarasu11

conda activate sundial
export EXPERIMENT_NAME=all224_glkn
export EXPERIMENT_SUFFIX=unet3dmid_dice_embed
export BASE_PATH=/home/ceoas/truongmy/emapr/sundial/utils/glkn/
#experiment_suffixes=("embed_pretrained_2" "no_pretraining" "setr_pup_dice_2_nockpt" "setr_pup_dice_2_unfrozen" "unet3d_dice")
#experiment_suffixes=("pretrained2_embed" "random_init_embed" "fcn_dice_unfrozen_embed" "fcn_dice_nockpt_embed" "unet3d_dice_embed", "unet3dmid_dice_embed")

srun python /home/ceoas/truongmy/emapr/sundial/utils/glkn/embeddings_00.py && \
srun python /home/ceoas/truongmy/emapr/sundial/utils/glkn/embeddings_01.py && \
srun python /home/ceoas/truongmy/emapr/sundial/utils/glkn/embeddings_02.py