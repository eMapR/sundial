#!/bin/bash -l
#experiment_suffixes=("embed_pretrained_2" "no_pretraining" "setr_pup_dice_2_nockpt" "setr_pup_dice_2_unfrozen" "unet3d_dice")

#experiment_suffixes=('300m_tl_all_embed' '300m_tl_notime_embed' '300m_tl_noboth_embed' '300m_tl_noloca_embed' '300m_all_embed' \
experiment_suffixes=('300m_tl_all_nrs_embed' '300m_tl_notime_nrs_embed' '300m_tl_noboth_nrs_embed' '300m_tl_noloca_nrs_embed' '300m_all_nrs_embed' \
                     '300m_fcn_dice_nockpt_nrs_embed' '300m_fcn_dice_unfrozen_nrs_embed' )

export EXPERIMENT_NAME="all224_glkn"
export BASE_PATH="/home/ceoas/truongmy/emapr/sundial/adhoc_analysis/embeddings/"

for suffix in "${experiment_suffixes[@]}"; do
    mkdir -p $BASE_PATH/$suffix
    sbatch \
        --export=EXPERIMENT_NAME,BASE_PATH,EXPERIMENT_SUFFIX="${suffix}"\
        --job-name=${suffix}\
        --output=${BASE_PATH}/${suffix}.log\
        --error=${BASE_PATH}/${suffix}.err\
        /home/ceoas/truongmy/emapr/sundial/adhoc_analysis/embeddings/embeddings_00_02.slurm
done