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
python -m showmak3r.main.train_actor \
-s ${DATA_PATH}   -r 1 \
--background_path ${DATA_PATH}/composite \
--mask_path ${DATA_PATH}/composite/actor_masks \
--foreground_mask_path ${DATA_PATH}/composite/foreground_masks \
--data_device cpu \
--model_path ${MODEL_PATH} \
--opt_smpl \
--sh_degree 3 \
--exp_name ${EXP_NAME} \
--eval_with_black_bg \
--iterations 10490 \
--iter_clip_person_shs 10 \
--clip_init_smpl_opacity \
--no_smpl_view_dir_reg \
--use_lpips_loss \
--use_density_reg_loss \
--lambda_init_smpl_verts_reg 0. \
--use_diffusion_guidance \
--textual_inversion_path ${MODEL_PATH}/diffusion \
--textual_inversion_method ${EXP_NAME} \
--use_ti_free_prompt_on_controlnet \
--dgm_noise_sched time_annealing \
--dgm_camera_sched defacto \
--dgm_start_iter 0 \
--lambda_rgb_loss 1000000 \
--dgm_cfg_scale 50 \
--use_adaptive_rgb_loss \
--dgm_cfg_rescale_weight 0. \
--use_mask_loss \
--random_background \
--use_deform \
--reverse_mask \
--use_wandb