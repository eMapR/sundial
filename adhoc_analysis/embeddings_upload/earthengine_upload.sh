#!/bin/bash -l

# Loop through numbers 0 to 7
for i in {0..6}; do
    INDEX=$(printf "%02d" $i)
    ASSET_ID="projects/pc464-mas-fvs/assets/emapr/300m_224_${INDEX}"
    FILE_PATH="gs://emapr/300m_224/*_t${INDEX}*"
    echo $FILE_PATH
    echo $ASSET_ID
    earthengine upload image --asset_id="$ASSET_ID" "$FILE_PATH"
done