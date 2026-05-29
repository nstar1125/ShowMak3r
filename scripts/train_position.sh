#!/bin/sh
set -e
. ~/miniconda3/etc/profile.d/conda.sh # conda sourcing -> change to your anaconda directory

DATA_NAME=$1
GPU_ID=$2

DATA_PATH=$(pwd)/demo/${DATA_NAME}
MODEL_PATH=$(pwd)/results/${DATA_NAME}

conda activate sm3r
python -m showmak3r.main.train_position \
    --data_path ${DATA_PATH} \
    --model_path ${MODEL_PATH} \
    --gpu ${GPU_ID}