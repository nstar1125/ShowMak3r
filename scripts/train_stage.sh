#!/bin/sh
set -e
. ~/miniconda3/etc/profile.d/conda.sh # conda sourcing -> change to your anaconda directory

DATA_NAME=$1
GPU_ID=$2

DATA_PATH=$(pwd)/demo/${DATA_NAME}
OUTPUT_PATH=$(pwd)/results/${DATA_NAME}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

conda activate sm3r
# for TV shows
python -m showmak3r.main.train_stage \
   -s ${DATA_PATH} \
   -r 1 \
   --background_path ${DATA_PATH}/composite \
   --iterations 10000 \
   --data_device cuda \
   --model_path ${OUTPUT_PATH}/stage \
   --sh_degree 3 \
   --random_background

# # for Web videos, set iterations to 3000
# python -m showmak3r.main.train_stage \
#    -s ${DATA_PATH} \
#    -r 1 \
#    --background_path ${DATA_PATH}/composite \
#    --iterations 3000 \
#    --data_device cuda \
#    --model_path ${OUTPUT_PATH}/stage \
#    --sh_degree 3 \
#    --random_background