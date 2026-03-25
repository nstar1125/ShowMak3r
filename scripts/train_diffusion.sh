#!/bin/bash
set -e
. ~/miniconda3/etc/profile.d/conda.sh # conda sourcing -> change to your anaconda directory

DATA_NAME=$1
GPU_ID=$2

DATA_PATH=$(pwd)/demo/${DATA_NAME}
MODEL_PATH=$(pwd)/results/${DATA_NAME}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export NCCL_P2P_DISABLE=1

conda activate sm3r

# 1. Prepare textual inversion
python -m showmak3r.main.train_diffusion \
    -s ${DATA_PATH} -r 1 \
    --mask_path ${DATA_PATH} \
    --data_device cpu \
    --model_path ${MODEL_PATH} \
    --use_inpaint_sd_for_masked_images \
    --crop_mode inpaint \
    --cd_use_controlnet \
    --cd_controlnet_mode v2 \
    --gen_mask

# 2. Train Custom Diffusion (with accelerate)
accelerate launch --main_process_port 8080 \
    -m showmak3r.main.train_diffusion \
    -s ${DATA_PATH} -r 1 \
    --mask_path ${DATA_PATH} \
    --data_device cpu \
    --model_path ${MODEL_PATH} \
    --use_inpaint_sd_for_masked_images \
    --cd_use_controlnet \
    --cd_controlnet_mode v2 \
    --use_ti_free_prompt_on_controlnet \
    --cd_bg_loss_weight 0.5 \
    --cd_random_bg \
    --cd_use_view_dependent_prompt \
    --cd_get_img_wo_resize \
    --cd_controlnet_weight 0.7 \
    --optimize_cd