#!/bin/bash -l

earthengine create collection projects/pc464-mas-fvs/assets/emapr/3cls_glkn_unet_full_aoi_merged
for year in $(seq 1994 2019); do
    filename="${year}.tif"
    ASSET_ID="projects/pc464-mas-fvs/assets/emapr/3cls_glkn_unet_full_aoi_merged/${year}"
    FILE_PATH="gs://emapr/3cls_glkn_unet_full_aoi_merged/3cls_glkn_unet_full_aoi_merged/${filename}"
    echo $FILE_PATH
    echo $ASSET_ID
    earthengine upload image --asset_id="$ASSET_ID" "$FILE_PATH"
done