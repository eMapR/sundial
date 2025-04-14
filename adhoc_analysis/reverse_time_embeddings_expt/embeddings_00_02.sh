#!/bin/bash -l

# these are identical instances of inference as below with some reshaping before export for the purpose of sanity checking
# experiment_suffixes=('300m_tl_all_embed' '300m_tl_notime_embed' '300m_tl_noboth_embed' '300m_tl_noloca_embed' '300m_all_embed')
experiment_suffixes=('300m_tl_all_nrs_embed' \
                    '300m_tl_notime_nrs_embed' \
                    '300m_tl_noboth_nrs_embed' \
                    '300m_tl_noloca_nrs_embed' \
                    '300m_all_nrs_embed' \
                    '300m_fcn_dice_nockpt_nrs_embed'
                    '300m_fcn_dice_unfrozen_nrs_embed')

export BASE_PATH="/home/ceoas/truongmy/emapr/sundial/adhoc_analysis/reverse_time_embeddings_expt/"
export EXPERIMENT_NAME="all224_glkn"

for suffix in "${experiment_suffixes[@]}"; do
    mkdir -p $BASE_PATH/$suffix
    mkdir -p $BASE_PATH/logs
    sbatch \
        --export=EXPERIMENT_NAME,BASE_PATH,EXPERIMENT_SUFFIX="${suffix}"\
        --job-name=${suffix}\
        --output=${BASE_PATH}/logs/${suffix}.log\
        --error=${BASE_PATH}/logs/${suffix}.err\
        /home/ceoas/truongmy/emapr/sundial/adhoc_analysis/reverse_time_embeddings_expt/embeddings_00_02.slurm
done