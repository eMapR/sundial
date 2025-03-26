#!/bin/bash -l
#SBATCH --job-name=collate_mosaic_56
#SBATCH --output=/home/ceoas/truongmy/emapr/sundial/adhoc_analysis/embeddings_upload/collate_mosaic_56.log
#SBATCH --error=/home/ceoas/truongmy/emapr/sundial/adhoc_analysis/embeddings_upload/collate_mosaic_56.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000
#SBATCH --partition=ceoas-arm
#SBATCH --nodelist=youmu

BASE_PATH="/home/ceoas/truongmy/emapr/sundial/predictions/emb_glkn/300m_56"
OUT_PATH="/home/ceoas/truongmy/emapr/sundial/predictions/emb_glkn/300m_56_proc"
PATTERN="*_t??_embed.pt"

mkdir -p "$OUT_PATH"

conda activate ${SUNDIAL_ENV_NAME}
srun --exclusive python /home/ceoas/truongmy/emapr/sundial/adhoc_analysis/embeddings_upload/collate_mosaic.py \
    "$BASE_PATH" \
    "$OUT_PATH" \
    "$PATTERN" \
    338 \
    224 \
    56 \
    1 \
    1 \
    $SLURM_PROCID