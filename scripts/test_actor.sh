#!/bin/sh
set -e
. ~/miniconda3/etc/profile.d/conda.sh # conda sourcing -> change to your anaconda directory

DATA_NAME=$1
EXP_NAME=$2
GPU_ID=$3

DATA_PATH=$(pwd)/demo/${DATA_NAME}
MODEL_PATH=$(pwd)/results/${DATA_NAME}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

conda activate sm3r
python -m showmak3r.main.test_actor \
    -s ${DATA_PATH}   -r 1 \
    --background_path ${DATA_PATH}/composite \
    --mask_path ${DATA_PATH}/composite/actor_masks \
    --foreground_mask_path ${DATA_PATH}/composite/foreground_masks \
    --model_path ${MODEL_PATH} \
    --exp_name ${EXP_NAME} \
    --white_background \
    --use_deform \
    --render
    # --viser
    