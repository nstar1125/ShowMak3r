#!/bin/sh
set -e
. ~/miniconda3/etc/profile.d/conda.sh # conda sourcing -> change to your anaconda directory

DATA_NAME=$1
GPU_IDS=$2

DATA_PATH=$(pwd)/demo/${DATA_NAME}

conda activate sm3r

python -m scripts.prep.prep_video \
    --data ${DATA_NAME} \
    --gpus ${GPU_IDS}

python -m scripts.prep.prep_composite_mono \
    --data ${DATA_NAME} \
    --gpus ${GPU_IDS}