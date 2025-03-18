#!/bin/bash -l
experiment_suffixes=("embed_pretrained_2" "no_pretraining" "setr_pup_dice_2_nockpt" "setr_pup_dice_2_unfrozen" "unet3d_dice")

export EXPERIMENT_NAME="all224_glkn"
export BASE_PATH="/home/ceoas/truongmy/emapr/sundial/utils/glkn/"

for suffix in "${experiment_suffixes[@]}"; do
    sbatch \
        --export=EXPERIMENT_NAME,BASE_PATH,EXPERIMENT_SUFFIX="${suffix}"\
        --job-name=${suffix}\
        --output=${BASE_PATH}/${suffix}.log\
        --error=${BASE_PATH}/${suffix}.err\
        /home/ceoas/truongmy/emapr/sundial/utils/glkn/embeddings_00_02.slurm
done