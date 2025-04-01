#!/bin/bash -l
#SBATCH --job-name=collate_mosaic_112
#SBATCH --output=/home/ceoas/truongmy/emapr/sundial/adhoc_analysis/embeddings_upload/logs/collate_mosaic_112.log
#SBATCH --error=/home/ceoas/truongmy/emapr/sundial/adhoc_analysis/embeddings_upload/logs/collate_mosaic_112.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128000
#SBATCH --partition=ceoas-arm
#SBATCH --nodelist=youmu

BASE_PATH="/home/ceoas/truongmy/emapr/sundial/predictions/emb_glkn/300m_112"
OUT_PATH="/home/ceoas/truongmy/emapr/sundial/predictions/emb_glkn/300m_112_proc"
PATTERN="*_t??_embed.pt"

mkdir -p "$OUT_PATH"

conda activate ${SUNDIAL_ENV_NAME}
srun --exclusive python /home/ceoas/truongmy/emapr/sundial/adhoc_analysis/embeddings_upload/collate_mosaic.py \
    "$BASE_PATH" \
    "$OUT_PATH" \
    "$PATTERN" \
    28 \
    14 \
    7 \
    1 \
    1 \
    $SLURM_PROCID