#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import importlib
numpy = importlib.import_module('numpy')
numpy.float = numpy.float32
numpy.int = numpy.int32
numpy.bool = numpy.bool_
numpy.unicode = numpy.unicode_
numpy.complex = numpy.complex_
numpy.object = numpy.object_
numpy.str = numpy.dtype.str


import os
import sys
import numpy as np
import shutil
import torch
import cv2
import copy
from typing import List, Union, NamedTuple, Any, Optional, Dict
from random import randint, random
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from math import floor, ceil
from PIL import Image
from diffusers import (
    StableDiffusionInpaintPipeline, 
    StableDiffusionPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel, 
    UniPCMultistepScheduler,
)
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from config.actor import ModelParams, PipelineParams, OptimizationParams, HumanOptimizationParams

from showmak3r.pipeline.dataset.composite_loader import load_composite_data
from showmak3r.pipeline.scene import Scene

from showmak3r.utils.general_utils import safe_state
from showmak3r.utils.image_utils import gen_videos, img_add_text
from showmak3r.utils.jnts_utils import filter_invisible_joints, extract_square_bbox
from showmak3r.utils.draw_op_jnts import draw_op_img

def get_face_crop_bbox(pj_jnts):
    valid_inds = [0, -4, -3, -2, -1]
    valid_jnts = pj_jnts[valid_inds]
    bbox_offset_ratio = 1.6
    bbox = extract_square_bbox(valid_jnts, offset_ratio=bbox_offset_ratio, get_square=True)

    return bbox
    
def get_crop_img_w_jnts(img, bbox, projected_jnts, rescale: float=1.2, resize: int=512):
    min_x = bbox[0]
    min_y = bbox[1]
    max_x = bbox[2]
    max_y = bbox[3]
    
    _w = int((max_x-min_x)*rescale)
    _h = int((max_y-min_y)*rescale)
    c_x = (min_x + max_x) // 2
    c_y = (min_y + max_y) // 2
    
    w = _w if _w>_h else _h
    h = w

    x = floor(c_x - w//2)
    y = floor(c_y - h//2)

    '''Crop in rectangular shape'''
    '''pad imgs when bbox is out of img'''
    x_front = 0   # offset for the case when we padded in front of the img.
    y_front = 0
    x_back = 0
    y_back = 0
    
    if x<0:
        x_front = -x
    if y<0:
        y_front = -y
    if x+w>= img.shape[1]:
        x_back = x+w-img.shape[1]+1
    if y+h>=img.shape[0]:
        y_back = y+w-img.shape[0]+1

    if x_front+y_front+x_back+y_back > 0:
        ext_img = cv2.copyMakeBorder(img, y_front, y_back, x_front, x_back, cv2.BORDER_CONSTANT, value=(0,0,0))
        x = x + x_front
        y = y + y_front
    else:
        ext_img = img
    cropped_img = ext_img[y:y+h, x:x+h]
    
    if isinstance(projected_jnts, List):
        _projected_jnts = []
        for _jnt in projected_jnts:
            if _jnt is None:
                _projected_jnts.append(_jnt)
            else:
                new_jnt = [0, 0]
                new_jnt[0] = _jnt[0] - (x - x_front)
                new_jnt[1] = _jnt[1] - (y - y_front)
                _projected_jnts.append(new_jnt)
        projected_jnts = _projected_jnts
    else:
        projected_jnts = projected_jnts - np.array([[x - x_front, y - y_front]])


    if resize > 0:
        re_cropped_img = cv2.resize(cropped_img, (resize, resize))
        scale_factor = resize / h
        
        if isinstance(projected_jnts, List):
            _projected_jnts = []
            for _jnt in projected_jnts:
                if _jnt is None:
                    _projected_jnts.append(_jnt)
                else:
                    new_jnt = [0, 0]
                    new_jnt[0] = (_jnt[0] - (h/2)) * scale_factor + resize/2
                    new_jnt[1] = (_jnt[1] - (h/2)) * scale_factor + resize/2
                    _projected_jnts.append(new_jnt)
            re_projected_jnts = _projected_jnts
        else:
            re_projected_jnts = (projected_jnts - np.array([[h/2, h/2]])) * scale_factor + np.array([[resize/2, resize/2]])
    
        return cropped_img, projected_jnts, re_cropped_img, re_projected_jnts
    else:
        return cropped_img, projected_jnts

def get_smallest_bbox(mask):
    # Find the coordinates of non-zero elements in the mask
    non_zero_coords  = np.argwhere(mask)
    if len(non_zero_coords) == 0:
        return None, None, None, None
    # assert len(non_zero_coords) != 0
    rows, cols = non_zero_coords[:, 0], non_zero_coords[:, 1]

    # Calculate the bounding box
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # Return the bounding box coordinates
    return min_row, min_col, max_row, max_col

def crop_img_with_mask(img, mask, pj_jnts, crop_mode: str='default', pipe=None, ti_pipe=False):
    alpha = (mask[...,None] * 255).astype(np.int8) 
    img = np.concatenate([img, alpha], axis=-1)


    if crop_mode == 'default':
        min_x, min_y, max_x, max_y = get_smallest_bbox(mask)    # need to invert as it's cv2 (y,x)
        
        if min_x is None:
            return None, None, None, None, None, None

        raw_cropped_img, raw_projected_jnts, cropped_img, projected_jnts = get_crop_img_w_jnts(img, [min_y, min_x, max_y, max_x], pj_jnts, rescale=1.1, resize=512)
        cropped_img = cropped_img.astype(np.uint8)
        raw_cropped_img = raw_cropped_img.astype(np.uint8)
        
        masked_imgs = cropped_img

        # projected_jnts = filter_invisible_joints(projected_jnts)
        joint_imgs = draw_op_img(projected_jnts, 512)
        joint_imgs = np.array(joint_imgs)[..., ::-1]
        raw_masked_img = cv2.resize(masked_imgs, (raw_cropped_img.shape[0], raw_cropped_img.shape[1])) 
    
    if crop_mode == 'face':
        min_x, min_y, max_x, max_y = get_smallest_bbox(mask)    # need to invert as it's cv2 (y,x)
        face_min_x, face_min_y, face_max_x, face_max_y = get_face_crop_bbox(pj_jnts)
        
        if min_x is None:
            return None, None, None, None, None, None

        # it should be somewhat shared one
        min_x = face_min_x if face_min_x > min_x else min_x
        min_y = face_min_y if face_min_y > min_y else min_y
        max_x = face_max_x if face_max_x < max_x else max_x
        max_y = face_max_y if face_max_y < max_y else max_y

        # get center points
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        side_length = (max_x - min_x) if (max_x - min_x) > (max_y - min_y) else (max_y - min_y)
        min_x = center_x - side_length / 2
        min_y = center_y - side_length / 2
        max_x = center_x + side_length / 2
        max_y = center_y + side_length / 2
        print([min_y, min_x, max_y, max_x])

        raw_cropped_img, raw_projected_jnts, cropped_img, projected_jnts = get_crop_img_w_jnts(img, [min_y, min_x, max_y, max_x], pj_jnts, rescale=1.1, resize=512)
        cropped_img = cropped_img.astype(np.uint8)
        raw_cropped_img = raw_cropped_img.astype(np.uint8)
        
        masked_imgs = cropped_img

        # projected_jnts = filter_invisible_joints(projected_jnts)
        joint_imgs = draw_op_img(projected_jnts, 512)
        joint_imgs = np.array(joint_imgs)[..., ::-1]
        raw_masked_img = cv2.resize(masked_imgs, (raw_cropped_img.shape[0], raw_cropped_img.shape[1]))  
        
    elif crop_mode == 'inpaint':
        generator = torch.manual_seed(0)
        
        min_x, min_y, max_x, max_y = get_smallest_bbox(mask)    # need to invert as it's cv2 (y,x)
        if min_x is None:
            return None, None, None, None, None, None
        
        

        raw_cropped_img, raw_projected_jnts, cropped_img, projected_jnts = get_crop_img_w_jnts(img, [min_y, min_x, max_y, max_x], pj_jnts, rescale=1.1, resize=512)
        cropped_img = cropped_img.astype(np.uint8)
        raw_cropped_img = raw_cropped_img.astype(np.uint8)

        # projected_jnts = filter_invisible_joints(projected_jnts)
        op_cond_img = draw_op_img(projected_jnts, 512)
        joint_imgs = np.array(op_cond_img)[..., ::-1]
        # Draw projected jnts here.

        sd_inpaint_img = cropped_img[...,:3][..., [2,1,0]]
        sd_inpaint_img = Image.fromarray(sd_inpaint_img)
        sd_inpaint_mask = 255 - cropped_img[...,3]
        sd_inpaint_mask = Image.fromarray(sd_inpaint_mask)

        with torch.no_grad():
            if ti_pipe:
                prompts = "a photo of a <new1> person"
            else:
                prompts = "a photo of a person"
            

            image = pipe(
                prompt="a photo of a person",
                negative_prompt="art, generated, six fingers, unrealistic, cartoon",
                generator=generator,
                image=sd_inpaint_img,
                mask_image=sd_inpaint_mask,
                control_image=op_cond_img,
                num_inference_steps=20,
                # strength=args.strength,
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.,
            ).images[0]
            
        # Save Mask Concatentated Images
        masked_imgs = np.array(image)[..., ::-1]
        masked_imgs = np.concatenate([masked_imgs, cropped_img[...,-1:]], axis=-1)
        raw_masked_img = cv2.resize(masked_imgs, (raw_cropped_img.shape[0], raw_cropped_img.shape[1]))
        
    return masked_imgs, joint_imgs, cropped_img, raw_masked_img, raw_projected_jnts, raw_cropped_img

def load_inpainting_diffusion(use_inpaint_sd: bool=False, use_controlnet: bool=False, use_ti_on_inpaint: bool=False, ti_path=None):
    if use_inpaint_sd:
        torch_dtype = torch.float32
        if use_controlnet:
            controlnet_op = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch_dtype)
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", 
                controlnet=controlnet_op,
                torch_dtype=torch_dtype,
                safety_checker=None,
                local_files_only=False,
            )
        else:
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", 
                torch_dtype=torch_dtype,
                safety_checker=None,
                local_files_only=False,
            )
            
    else:
        torch_dtype = torch.float16
        if use_controlnet:
            controlnet_op = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch_dtype)
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                controlnet=controlnet_op,
                torch_dtype=torch_dtype,
                safety_checker=None,
                local_files_only=False,
            )
        else:
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                torch_dtype=torch_dtype,
                safety_checker=None,
                local_files_only=False,
            )
    # disable progress bar
    pipe.set_progress_bar_config(disable=True)
    
    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    if use_ti_on_inpaint:
        print("LOADING pretrained textual inversion!")
        pipe.unet.load_attn_procs(
            Path(ti_path), weight_name="pytorch_custom_diffusion_weights.bin"
        )
        pipe.load_textual_inversion(Path(ti_path), weight_name="<new1>.bin")
        print(f"[LOAD] Loading TI from {str(ti_path)}")

    return pipe
        
def train_ti(dataset, pipe, args):
    # Human train settings (change settings if needed here)
    human_train_opt = HumanOptimizationParams()
    human_train_opt.sh_degree = dataset.human_sh_degree
    human_train_opt.view_dir_reg = dataset.smpl_view_dir_reg

    # ------------------------- Load train datasets -------------------------
    scene, _, people_infos = \
        load_composite_data(
            dataset=dataset,
            pipe=pipe,
            type="ti",
            iteration=-1,
            exp_name=None,
            human_train_opt=human_train_opt
        )
    
    save_dir = Path(dataset.model_path) / "diffusion"
    save_dir.mkdir(exist_ok=True)

    image_dir = Path(dataset.source_path) / "video" / "frames"
    assert image_dir.exists()
    actor_dir = Path(dataset.model_path) / "actor"
    assert actor_dir.exists()

    # ============================== Prepare dataset ==============================
    if args.gen_mask:
        print(f"[INFO] Start dataset preparation")
        scene_cameras = scene.getTrainCameras()

        # start preparation
        for person_info in people_infos:
            print(f"[INFO] Preparing for person {person_info.person_number}")
            pnum = person_info.person_number
            person_save_dir = save_dir / pnum
            person_save_dir.mkdir(exist_ok=True)

            # get valid fnames for current person
            valid_fnames = []
            for cam in person_info.human_scene.getTrainCameras():
                valid_fnames.append(cam.fname)    

            mask_save_dir = person_save_dir / 'masked_images'
            mask_save_dir.mkdir(exist_ok=True)
            raw_mask_save_dir = person_save_dir / 'masked_images_raw'
            raw_mask_save_dir.mkdir(exist_ok=True)
            op_jnts_dir = person_save_dir / 'op_jnts'
            op_jnts_dir.mkdir(exist_ok=True)
            view_prompts_dir = person_save_dir / 'view_prompts'
            view_prompts_dir.mkdir(exist_ok=True)
            mask_jpg_save_dir = person_save_dir / 'masked_images_jpg'
            mask_jpg_save_dir.mkdir(exist_ok=True)
            op_cond_dir = person_save_dir / 'openpose_conditions'
            op_cond_dir.mkdir(exist_ok=True)
            op_overlay_dir = person_save_dir / 'op_overlay'
            op_overlay_dir.mkdir(exist_ok=True)
            align_overlay_dir = person_save_dir / 'align_overlay'
            align_overlay_dir.mkdir(exist_ok=True)
            
            # load inpainting module 
            if args.crop_mode == 'default' or args.crop_mode == 'face':
                inpaint_pipe = None
            elif args.crop_mode == 'inpaint':
                inpaint_pipe = load_inpainting_diffusion(
                                    args.use_inpaint_sd_for_masked_images, 
                                    True, 
                                    args.two_stage_inpaint, 
                                    actor_dir / pnum
                                    )
            else:
                raise NotImplementedError(f"Crop mode {args.crop_mode} is not implemented")
            
            # start preparation
            for cam in tqdm(scene_cameras, desc=f"Preparing - person {pnum}"):
                cam_fname = cam.fname
                if cam_fname in valid_fnames:
                    # load image and mask
                    gt_img_path = image_dir / f"{cam_fname}.png"
                    gt_mask_path = actor_dir / f"{int(pnum):03d}" / "masks" / f"{cam_fname}.png"
                    
                    gt_img = cv2.imread(str(gt_img_path))
                    gt_img = cv2.resize(gt_img, (cam.image_width, cam.image_height))

                    if gt_mask_path.exists():
                        gt_mask = (cv2.imread(str(gt_mask_path), 0)>1).astype(np.uint8)
                    else:
                        gt_mask = np.zeros((cam.image_height, cam.image_width), dtype=np.uint8)
                    gt_mask = cv2.resize(gt_mask, (cam.image_width, cam.image_height))
                    
                    raw_gt_img = gt_img.copy()
                    
                    if args.alpha_matt_pixels > 0:
                        # apply smoothing on mask 
                        # We define it in "carving way" as lack of information is acceptible while false positive is big defection.
                        
                        width_of_smoothing = int(args.alpha_matt_pixels)
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width_of_smoothing, width_of_smoothing))

                        # Perform erosion
                        gt_mask = cv2.erode(gt_mask, kernel, iterations=1)

                        # Apply Gaussian blur for smoother edges
                        gt_mask = cv2.GaussianBlur(gt_mask, (width_of_smoothing, width_of_smoothing), 0)
                    

                    # load projected_joints and prompts
                    _data_idx = person_info.fnames.index(cam_fname)
                    pj_jnts = person_info.misc['projected_op_jnts'][_data_idx]
                    view_prompts = person_info.misc['body_prompts'][_data_idx]

                    # Here we use BLACK background
                    gt_img[gt_mask==0] *= 0

                    # crop image based on mask
                    if len(gt_mask.shape) == 3:
                        gt_mask = (gt_mask.sum(-1) > 0)

                    # Get masked images
                    masked_img, joint_imgs, cropped_img, raw_masked_img, raw_projected_jnts, raw_cropped_img = \
                        crop_img_with_mask(gt_img, gt_mask, pj_jnts, 
                                           crop_mode = args.crop_mode, pipe=inpaint_pipe, 
                                           ti_pipe=args.use_inpaint_sd_for_masked_images)
                    if masked_img is None:
                        continue
                    
                    # save results
                    np.save(str(op_jnts_dir / f"{cam_fname}.npy"), raw_projected_jnts, allow_pickle=True)
                    np.save(str(view_prompts_dir / f"{cam_fname}.npy"), view_prompts, allow_pickle=True)
                    cv2.imwrite(str(raw_mask_save_dir / f"{cam_fname}.png"), raw_masked_img)
                    cv2.imwrite(str(mask_save_dir / f"{cam_fname}.png"), masked_img)
                    cv2.imwrite(str(mask_jpg_save_dir / f"{cam_fname}.jpg"), masked_img)
                    cv2.imwrite(str(op_cond_dir / f"{cam_fname}.png"), joint_imgs)
                    
                    # Debug Alignment
                    jnts_fg = (joint_imgs.sum(-1, keepdims=True) > 0)
                    img_fg = cropped_img[..., :3] * (1-jnts_fg) + joint_imgs * jnts_fg
                    cv2.imwrite(str(op_overlay_dir / f"{cam_fname}.jpg"), img_fg)
                    
                    # Debug Renderer
                    raw_jnts_img = draw_op_img(pj_jnts, raw_gt_img.shape[:2])
                    raw_jnts_img = np.array(raw_jnts_img)[..., ::-1]
                    jnts_fg = (raw_jnts_img.sum(-1, keepdims=True) > 0)
                    img_fg = raw_gt_img[..., :3] * (1-jnts_fg) + raw_jnts_img * jnts_fg
                    
                    img_fg = img_fg.astype(np.uint8)
                    img_fg = img_add_text(img_fg.copy(), f"prompts: {view_prompts}")
                    cv2.imwrite(str(align_overlay_dir / f"{cam_fname}.jpg"), img_fg)

            # remove inpaint model
            if args.crop_mode == 'inpaint':
                del inpaint_pipe

    # ============================== Train Custom Diffusion ==============================
    if args.optimize_cd:
        print(f"[INFO] Start Custom Diffusion training")
        from showmak3r.pipeline.diffusion.train_custom_diffusion import load_default_train_opt, train_cd

        for person_info in people_infos: # run train_custom_diffusion.py for each person
            # setup directories
            pnum = person_info.person_number
            person_save_dir = save_dir / pnum
            person_save_dir.mkdir(exist_ok=True)

            mask_save_dir = person_save_dir / 'masked_images'
            op_cond_dir = person_save_dir / 'openpose_conditions'
            
            # setup training options
            cd_train_opt = load_default_train_opt(args.model_name)
            cd_train_opt.instance_data_dir = str(mask_save_dir)
            cd_train_opt.instance_cond_dir = str(op_cond_dir)
            cd_train_opt.output_dir = str(person_save_dir)
            cd_train_opt.class_data_dir = "showmak3r/pipeline/diffusion/sample_person_photo"
            cd_train_opt.class_prompt = ""      # "human"
            cd_train_opt.num_class_images = 200
            cd_train_opt.instance_prompt = "photo of a <new1> person"  
            cd_train_opt.instance_prompt_wo_token = "photo of a person"  
            cd_train_opt.resolution = 512
            cd_train_opt.train_batch_size = args.cd_batch_size   # when using WITH PRIOR -> bsize 2 raise ERRORS
            cd_train_opt.learning_rate = args.cd_lrs             # default: 1e-5, 1e-6 is more specific version for face optimization
            cd_train_opt.lr_warmup_steps = 0
            cd_train_opt.max_train_steps = args.cd_steps
            cd_train_opt.scale_lr = True
            cd_train_opt.hfip = True                    
            cd_train_opt.modifier_token = "<new1>"
            cd_train_opt.validation_prompt = "photo of a <new1> person"
            cd_train_opt.report_to = "wandb"
            cd_train_opt.no_safe_serialization = True      
            cd_train_opt.use_controlnet = args.cd_use_controlnet   
            cd_train_opt.use_color_jitter = args.cd_use_color_jitter
            cd_train_opt.controlnet_mode = args.cd_controlnet_mode
            cd_train_opt.controlnet_weight = args.cd_controlnet_weight
            cd_train_opt.save_intermediate_for_debug = True     
            cd_train_opt.noaug = args.cd_no_aug
            cd_train_opt.bg_loss_weight = args.cd_bg_loss_weight
            cd_train_opt.random_bg = args.cd_random_bg
            cd_train_opt.image_space_loss = args.cd_loss_in_image_space
            cd_train_opt.loss_in_original_img_resolution = args.cd_loss_in_raw_resolution
            cd_train_opt.get_img_wo_resize = args.cd_get_img_wo_resize
            cd_train_opt.mask_cond_image_with_data_mask = args.cd_mask_controlnet_input
            
            if args.cd_get_img_wo_resize :
                op_cond_dir = person_save_dir / 'op_jnts'
                cd_train_opt.instance_cond_dir = str(op_cond_dir)
                mask_save_dir = person_save_dir / 'masked_images_raw'
                cd_train_opt.instance_data_dir = str(mask_save_dir)
                cd_train_opt.masking_cond_image = args.cd_masking_cond_image
                
            if args.cd_use_view_dependent_prompt :
                view_prompt_dir = person_save_dir / 'view_prompts'
                cd_train_opt.view_prompt_dir = str(view_prompt_dir)
            
            # Settings to save VRAMs
            cd_train_opt.enable_xformers_memory_efficient_attention = True
            cd_train_opt.set_grads_to_none = True

            if args.cd_use_prior:
                cd_train_opt.with_prior_preservation = True
                cd_train_opt.real_prior = True
                cd_train_opt.fullbody_prior = False
                cd_train_opt.prior_loss_weight = 1.
                cd_train_opt.prior_batch_size = args.cd_prior_batch_size
                
            if args.cd_use_fullbody_prior:
                cd_train_opt.with_prior_preservation = True
                cd_train_opt.real_prior = True
                cd_train_opt.fullbody_prior = True
                cd_train_opt.fullbody_prior_path = args.fullbody_prior_path
                cd_train_opt.prior_loss_weight = 1.
                cd_train_opt.prior_batch_size = args.cd_prior_batch_size

            if args.cd_controlnet_mode == "v4":
                raise NotImplementedError()
                cd_train_opt.cd_only_on_controlnet = False

            # ------------------------------ run training ------------------------------
            '''
            Train Custom Diffusion code from diffusers example
            https://github.com/huggingface/diffusers/blob/main/examples/custom_diffusion/train_custom_diffusion.py
            '''
            wandb_exp_name = f"{Path(dataset.model_path).name}_{pnum}"
            train_cd(cd_train_opt, wandb_exp_name)

            # ------------------------------ generate videos ------------------------------
            # Make video of optimization progress
            gen_videos([person_save_dir / 'optim_logs'], is_jpg=True, fps=10, rm_dir=False, regex_fname="0_*.jpg", save_tag="_0")
            gen_videos([person_save_dir / 'optim_logs'], is_jpg=True, fps=10, rm_dir=True, regex_fname="1_*.jpg", save_tag="_1")

            if cd_train_opt.save_intermediate_for_debug:
                if cd_train_opt.with_prior_preservation:
                    gen_videos([person_save_dir / 'debug_logs'], is_jpg=True, fps=30, rm_dir=False, regex_fname="priors_*.jpg", save_tag="_prior")
                gen_videos([person_save_dir / 'debug_logs'], is_jpg=True, fps=30, rm_dir=True, regex_fname="main_*.jpg", save_tag="_main")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # Basic options
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--gender', type=str, default='neutral')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)    
    parser.add_argument("--quiet", action="store_true")

    # Define process options
    parser.add_argument("--gen_mask", action='store_true')
    parser.add_argument("--optimize_cd", action='store_true')

    # For generating masked images
    parser.add_argument("--crop_mode", type=str, default="default")
    parser.add_argument("--two_stage_inpaint", action='store_true', help='use two-staged inpainting')
    # parser.add_argument("--inpaint_ti_dir", type=str, default="output_common")
    parser.add_argument("--alpha_matt_pixels", default=5, help='ratio of matting pixels considering ')

    # Diffusion model options
    parser.add_argument("--use_inpaint_sd_for_masked_images", action='store_true')
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")

    # CustomDiffusion train options
    parser.add_argument("--cd_batch_size", type=int, default=2)
    parser.add_argument("--cd_steps", type=int, default=1000)
    parser.add_argument("--cd_lrs", type=float, default=5e-6)
    parser.add_argument("--cd_use_prior", action='store_true')
    parser.add_argument("--cd_use_controlnet", action='store_true', help='whether using controlnet on CD training')
    parser.add_argument("--cd_controlnet_weight", type=float, default=0.8)
    parser.add_argument("--cd_use_inpainting_diffusion", action='store_true', help='whether using inpainting diffusion on training')
    parser.add_argument("--cd_use_color_jitter", action='store_true', help='turn on color jittering of CD')
    parser.add_argument("--cd_controlnet_mode", type=str, default="v1")
    parser.add_argument("--cd_no_aug", action='store_true', help='turn off spatial augmentations of CD')
    parser.add_argument("--cd_bg_loss_weight", type=float, default=1.)
    parser.add_argument("--cd_random_bg", action='store_true', help='turn ON that randomly select bg from white/black')
    parser.add_argument("--cd_loss_in_image_space", action='store_true', help='Apply loss after VAE decoder')
    parser.add_argument("--cd_loss_in_raw_resolution", action='store_true', help='Apply loss in original image resolution')
    parser.add_argument("--cd_use_fullbody_prior", action='store_true', help='whether useing fullbody prior for optimization')
    parser.add_argument("--cd_get_img_wo_resize", action='store_true', help='whether getting condition image with better approach')
    parser.add_argument("--cd_prior_batch_size", default=1)
    parser.add_argument("--cd_masking_cond_image", action='store_true', help='whether getting condition image with better approach')
    parser.add_argument("--fullbody_prior_path", type=str, default=None)
    parser.add_argument("--cd_use_view_dependent_prompt", action='store_true', help='Use view-dependent prompt conditioning')
    parser.add_argument("--cd_mask_controlnet_input", action='store_true', help='Mask controlnet input with data mask')
    
    args = parser.parse_args(sys.argv[1:])


    lp_extracted = lp.extract(args)
    lp_extracted.eval = False
    
    if args.cd_controlnet_mode not in [
        "v0",       # TI token disconnected on ControlNet
        "v1",       # All connected
        "v2",       # No TI token on ControlNet
        "v3",       # All Connected. Use aligned Conditioning.
        "v4",       # Apply TI token ONLY on ControlNet (+ FineTune ControlNet)
    ]:
        raise TypeError(f"'{args.cd_controlnet_mode}' is invalid mode")
            
    if args.cd_batch_size > 1 and args.cd_loss_in_image_space:
        print("\n[INFO] Reducing Batch size to avoid OOM error from image space")
        args.cd_batch_size = int(args.cd_batch_size/2)
        
    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    train_ti(lp_extracted, pp.extract(args), args)
    
    print("\nTraining complete.")
