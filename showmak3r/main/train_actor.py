import importlib
numpy = importlib.import_module('numpy')
numpy.float = numpy.float32
numpy.int = numpy.int32
numpy.bool = numpy.bool_
numpy.unicode = numpy.unicode_
numpy.complex = numpy.complex_
numpy.object = numpy.object_
numpy.str = numpy.dtype.str

import time
import os
import sys
import shutil
import torch
import pandas
import cv2
import numpy as np

from tqdm import tqdm
from omegaconf import OmegaConf
from typing import List, Union, NamedTuple, Any, Optional, Dict
from random import randint, random
from argparse import ArgumentParser
from pathlib import Path

from config.actor import ModelParams, PipelineParams, OptimizationParams, HumanOptimizationParams

from showmak3r.pipeline.renderer.gaussian_renderer import composite_render
from showmak3r.pipeline.renderer.diffusion_renderer import diffusion_renderer
from showmak3r.pipeline.renderer.renderer_wrapper import render_full_video, render_full_canonical, render_canonical_images, render_deform_images, render_log_images
from showmak3r.pipeline.dataset.composite_loader import load_composite_data
from showmak3r.pipeline.refine.deform_branch import DeformModel, get_residuals

from showmak3r.utils.general_utils import safe_state
from showmak3r.utils.loss_utils import l1_loss, ssim, denisty_reg_loss
from showmak3r.utils.io_utils import save_rgb_image
from showmak3r.utils.metric_utils import Evaluator

CLIP_HIGH_RENDERED_RGB = True

###########################################################################################
##                                   Training function                                   ##
###########################################################################################
def training(
    dataset, 
    opt, 
    pipe, 
    save_iterations, 
    test_iterations, 
    debug_from, 
    exp_name
    ):
    first_iter = 0

    # SMPL training related settings
    do_smpl_fitting_local = True    # fit SMPL pose during optimization
    do_smpl_fitting_global = False  # fit SMPL global orientation during optimization 
    
    # Human train settings
    human_train_opt = HumanOptimizationParams()
    human_train_opt.densify_from_iter = dataset.iter_smpl_densify - dataset.dgm_start_iter  # (start directly)
    human_train_opt.densify_until_iter = dataset.iter_densify_smpl_until
    human_train_opt.opacity_reset_interval = dataset.person_smpl_reset
    human_train_opt.view_dir_reg = dataset.smpl_view_dir_reg
    human_train_opt.sh_degree = dataset.human_sh_degree

    # SMPL clipping settings
    human_train_opt.clip_init_smpl_opacity = dataset.clip_init_smpl_opacity
    human_train_opt.smpl_opacity_clip_min = dataset.smpl_opacity_clip_min
    
    if human_train_opt.view_dir_reg:
        print("[INFO] view-direction regularizer is ON")

    # Load train datasets
    scene, scene_gaussians, people_infos = \
        load_composite_data(
            dataset=dataset,
            pipe=pipe,
            type="train", # ti, train, test
            iteration=-1,
            exp_name=exp_name,
            human_train_opt=human_train_opt,
        )
    
    for people_info in people_infos:
        if people_info.view_dir_reg:
            print(f"{people_info.person_number}: View-direction regularizer ON")
        else:
            print(f"{people_info.person_number}: View-direction regularizer OFF")

    if dataset.use_lpips_loss: # True
        from lpips import LPIPS
        lpips = LPIPS(net='vgg').cuda()

        # turn off grad tracking of lpips param
        nets = lpips
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = False
    
    # add refinement model
    if dataset.use_deform:
        deform = DeformModel()
        deform.train_setting(opt)
    else:
        deform = None
    
    # setup DGM module
    if dataset.use_diffusion_guidance:
        # we also use DGM module to sample camera similarly
        from showmak3r.pipeline.diffusion.guidance import DiffusionGuidance
        pnums = [pi.person_number for pi in people_infos]

        # Diffusion Guidance setup
        os.makedirs(scene.model_path + "/training", exist_ok=True)
        dg_log_dir = Path(scene.model_path) / "training" / "log_diffusion"
        dgm_opt = OmegaConf.load("showmak3r/pipeline/diffusion/guidance/configs/default.yaml")
        dgm_opt.density_start_iter = dataset.iter_smpl_densify
        dgm_opt.density_end_iter = dataset.iter_densify_smpl_until
        dgm_opt.densification_interval = human_train_opt.densification_interval
        dgm_opt.iter_prune_smpl_until = dataset.iter_prune_smpl_until
        dgm_opt.iter_smpl_densify = dataset.iter_smpl_densify
        dgm_opt.scene_extent = float(scene.cameras_extent)

        # Initialize DGM module
        textual_inversion_path = Path(dataset.textual_inversion_path) if dataset.textual_inversion_path != "" else None
        DGM = DiffusionGuidance(
            opt=dgm_opt, 
            log_dir=dg_log_dir, 
            textual_inversion_path=textual_inversion_path, 
            textual_inversion_in_controlnet=dataset.use_ti_in_controlnet,
            use_ti_free_prompt_on_controlnet = dataset.use_ti_free_prompt_on_controlnet,
            ti_load_epoch = dataset.ti_chkpt_epoch,
            guidance_scale = dataset.dgm_cfg_scale,
            inpaint_guidance_scale = dataset.dgm_inpaint_guidance_scale,
            controlnet_weight = dataset.dgm_controlnet_weight,
            lambda_percep=opt.lambda_dgm_percep,
            lambda_rgb=opt.lambda_dgm_rgb,
            random_noise_step = dataset.dgm_random_sample,
            noise_sched = dataset.dgm_noise_sched,
            camera_sched = dataset.dgm_camera_sched,
            do_guid_sched = dataset.dgm_guidance_decay,
            sd_version="1.5",
            use_aux_prompt = dataset.dgm_use_aux_prompt,
            use_view_prompt = dataset.dgm_use_view_prompt,
            cfg_sched = dataset.dgm_cfg_sched,
        )
        print("------------------------------------------------------------")
        print("\t[INFO] Start loading diffusion guidance module...")
        print("------------------------------------------------------------")
        # Prepare DGM module
        DGM.prepare_train(
            pnums, 
            enable_controlnet = dataset.dgm_enable_sdop,
            do_cfg_rescale = dataset.dgm_use_cfg_rescale,
        )
    else:
        DGM = None
    
    # background setup
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    black_bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    eval_bg = black_bg if dataset.eval_with_black_bg else white_bg
        
    # setup wandb
    if pipe.use_wandb:
        import wandb
        wandb.init(
            project="ShowMak3r_actor",
            name=exp_name
        )
        setattr(dataset, '_human_opts', human_train_opt)
        setattr(dataset, '_train_opts', opt)
        wandb.config.update(dataset)
        wandb.config.update(opt)

        parsed_dict = wandb.sdk.wandb_helper.parse_config(human_train_opt)
        renamed_dict = dict()
        for k, v in parsed_dict.items():
            renamed_dict[f"human_opt_{k}"] = v
        wandb.config.update(renamed_dict)
        
        # define our custom x axis metric
        wandb.define_metric("loss/step")
        wandb.define_metric("loss/*", step_metric="loss/step")

        # define time axis for metric
        wandb.define_metric("metric/*", step_metric="loss/step")
        wandb.define_metric("dg_loss/*", step_metric="loss/step")
        wandb.define_metric("infos/*", step_metric="loss/step")
        
        if dataset.use_diffusion_guidance:
            for pi in people_infos:
                wandb.define_metric(f"_{pi.person_number}/*", step_metric=f"loss/step")
                
        # Back up SMPL-scale
        log_dict = dict()
        for pi in people_infos:
            log_dict[f"_{pi.person_number}/smpl_scale"] = pi.smpl_scale.item()
        wandb.log(log_dict)

        # evaluator = Evaluator

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    front_mask_dict = dict()

    # ==============================  start training  ==============================
    print("------------------------------------------------------------")
    print("\t[INFO] Start training...")
    print("------------------------------------------------------------")
    for iteration in range(first_iter, opt.iterations + 1):  
        
        iter_start.record()
        
        # update learning rate for each person
        for person_info in people_infos:
            if dataset.use_diffusion_guidance:
                if iteration > dataset.dgm_start_iter:
                    person_info.gaussians.update_learning_rate(iteration)
            else:
                person_info.gaussians.update_learning_rate(iteration)

            if person_info.do_trans_grid:
                person_info.grid_optimizer.zero_grad()

            if do_smpl_fitting_local:
                person_info.local_pose_optimizer.zero_grad()

            if do_smpl_fitting_global:
                person_info.global_pose_optimizer.zero_grad()

        # enlarge SH degree for each person
        for person_info in people_infos:
            if (person_info.misc['optimized_step'] % dataset.human_sh_degree_enlarge_interval == 0) and (person_info.misc['optimized_step'] > 0):
                person_info.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        fname = viewpoint_cam.fname 

        if (iteration - 1) == debug_from:
            pipe.debug = True
            torch.autograd.set_detect_anomaly(True)
        
        # random background
        if dataset.random_background:
            background = torch.rand(3).float().cuda()

        # calculate refinement residuals
        total_frame = len(scene.getTrainCameras().copy())
        if iteration < dataset.start_deform or not dataset.use_deform:
            d_color, d_opacity = 0.0, 0.0
        else:
            d_color, d_opacity = get_residuals(people_infos, deform, fname, total_frame)
        # ==============================  Calculate reconstruction loss  ==============================

        if opt.lambda_rgb_loss > 0:
            # ------------------------- 1. Render and get GTs -------------------------
            # Render actor gaussians
            human_render_pkg = composite_render(viewpoint_cam, scene_gaussians, people_infos, pipe, background, d_color, d_opacity, \
                                            scaling_modifier = 1.0, override_color = None, render_only_people=True)
            if human_render_pkg == None:
                print(f"[INFO] No human in the frame... skipping iteration {iteration}")
                continue
            image, viewspace_point_tensor, visibility_filter, radii = \
                human_render_pkg["render"], human_render_pkg["viewspace_points"], human_render_pkg["visibility_filter"], human_render_pkg["radii"]
            # actor masks
            render_mask = human_render_pkg["render"]==background.view(3, 1, 1) 
            render_mask = torch.all(render_mask, axis=0).type(torch.uint8)
            render_mask = 1-render_mask.unsqueeze(0) # reverse mask
            # alpha value
            alpha = human_render_pkg['mask'].squeeze()
            
            # calculate front masks
            if iteration<total_frame+1: # keep first set of front masks
                composite_render_pkg = composite_render(viewpoint_cam, scene_gaussians, people_infos, pipe, background, d_color, d_opacity, \
                                                scaling_modifier = 1.0, override_color = None, render_only_people=False)
                composite_depth = composite_render_pkg["depth"]
                human_depth = human_render_pkg["depth"]
                hn_depth = human_depth*render_mask
                fore_depth = composite_depth*render_mask
                
                noise = 0.2
                front_mask = fore_depth < hn_depth - noise # keep unwanted background floaters masking
                front_mask_dict[fname] = ~front_mask
        
            # clip saturated points
            if CLIP_HIGH_RENDERED_RGB:
                image[image > 1] /= image[image > 1] 
            
            # get GTs
            gt_image = viewpoint_cam.gt_image.cuda()
            
            # ------------------------- 2. Apply masks -------------------------
            assert (viewpoint_cam.gt_mask is not None), f"[ERROR] GT alpha mask of '{str(viewpoint_cam.mask_fname)}' not exists"
            
            # apply GT mask
            gt_mask = viewpoint_cam.gt_mask.cuda()[:3, :, :]
            gt_mask = 1 - gt_mask # reverse mask
            if dataset.reverse_mask:
                gt_mask = 1 - gt_mask
            gt_image = gt_image * gt_mask + (1-gt_mask) * torch.ones_like(gt_image) * background[:,None,None]
            
            # apply front mask
            gt_image = gt_image*front_mask_dict[fname]
            image = image*front_mask_dict[fname]
            
            # apply foreground mask
            if (not scene.fmask_dict is None) and (fname in scene.fmask_dict):
                height = gt_image.shape[-2]
                width = gt_image.shape[-1]

                fore_mask = scene.fmask_dict[fname].cuda()[None, None]
                fore_mask = torch.nn.functional.interpolate(fore_mask, (height, width), mode="bilinear", align_corners=False)
                fore_mask = fore_mask[0]
                if dataset.reverse_mask:
                    fore_mask = 1 - fore_mask

                image = image * fore_mask
                gt_image = gt_image * fore_mask
                
            # ------------------------- 3. Calculate Losses -------------------------
            # Calculate L1 loss
            Ll1 = l1_loss(image, gt_image)
            
            # Calculate D-SSIM loss
            dssim_loss = (1.0 - ssim(image, gt_image))
            
            # Calculate LPIPS loss
            if dataset.use_lpips_loss:
                lpips_loss = lpips(image[None], gt_image[None]).mean()
            else:
                lpips_loss = 0
                
            # (Optional) if optimized with NR grid
            grid_trans_reg_loss = 0
            if person_info.do_trans_grid:
                for person_info in people_infos:
                    if fname not in person_info.fnames:
                        continue
                    grid_trans_reg_loss += (person_info.smpl_deformer.last_trans[-1] ** 2).mean()
                    grid_trans_reg_loss = grid_trans_reg_loss
                     
            # (Optional) if truned on caluclate density regularization loss
            density_reg = 0
            if dataset.use_density_reg_loss:
                density_reg = denisty_reg_loss(alpha)
                
            # (Optional) Calculate mask loss
            mask_loss = 0
            if dataset.use_mask_loss:
                mask_loss = l1_loss(alpha.squeeze(), gt_mask.squeeze())
            
            # total loss
            loss = (1.0 - opt.lambda_dssim) * Ll1 \
                    + opt.lambda_dssim * dssim_loss \
                    + opt.lambda_mask * mask_loss \
                    + opt.lambda_trans_reg * grid_trans_reg_loss \
                    + opt.lambda_lpips * lpips_loss
                    
            # get image_loss scaler
            lambda_rgb = opt.lambda_rgb_loss
            
            rgb_scaler = 1.
            if dataset.use_diffusion_guidance:
                # Apply strong recon loss ONLY after optimizing with DGM
                if (not (iteration > dataset.dgm_start_iter)) or (iteration % dataset.apply_dgm_every_n != 0):
                    rgb_scaler = 1 / opt.lambda_rgb_loss
                elif dataset.use_adaptive_rgb_loss:
                    dgm_step, max_noise_ratio = DGM.get_noise_level()
                    rgb_scaler = max_noise_ratio ** 2
                lambda_rgb = lambda_rgb * rgb_scaler
                
            loss = loss * lambda_rgb
            loss = loss + opt.lambda_density_reg * density_reg

            # regularize gaussians drifting to zero opacity
            if iteration >= dataset.start_deform and dataset.use_deform:
                loss += opt.lambda_opacity_reg * lambda_rgb * torch.mean(torch.relu(-d_opacity))
                loss += opt.lambda_color_reg * lambda_rgb * torch.mean(torch.abs(d_color))
            
            # ------------------------- 4. Logging -------------------------
            log_dict = dict()
            if pipe.use_wandb:
                log_dict = {
                    "loss/step": iteration,
                    "loss/dssim_loss": lambda_rgb * opt.lambda_dssim * dssim_loss.detach(),
                    "loss/rgb_loss": lambda_rgb * (1.0 - opt.lambda_dssim) * Ll1.detach(),
                    "loss/lpips_loss": lambda_rgb * opt.lambda_lpips * (lpips_loss.detach() if isinstance(lpips_loss, torch.Tensor) else float(lpips_loss)),
                    "loss/mask_loss": lambda_rgb * opt.lambda_mask * (mask_loss.detach() if isinstance(mask_loss, torch.Tensor) else float(mask_loss)),
                    "loss/grid_reg_loss": lambda_rgb * opt.lambda_trans_reg * (grid_trans_reg_loss.detach() if isinstance(grid_trans_reg_loss, torch.Tensor) else float(grid_trans_reg_loss)),
                    "loss/density_reg_loss": lambda_rgb * opt.lambda_density_reg * (density_reg.detach() if isinstance(density_reg, torch.Tensor) else float(density_reg)),
                    "infos/d_color": d_color,
                    "infos/d_opacity": d_opacity,
                }
                if dataset.use_diffusion_guidance:
                    log_dict["infos/lambda_rgb"] = lambda_rgb
                    log_dict["loss/raw_ssim_loss"] = opt.lambda_dssim * dssim_loss.detach() 
                    log_dict["loss/raw_rgb_loss"] = (1.0 - opt.lambda_dssim) * Ll1.detach()
                    log_dict["loss/raw_lpips_loss"] = opt.lambda_lpips *  (lpips_loss.detach() if isinstance(lpips_loss, torch.Tensor) else float(lpips_loss))
                    log_dict["loss/raw_mask_loss"] = opt.lambda_mask * (mask_loss.detach() if isinstance(mask_loss, torch.Tensor) else float(mask_loss)),
                    log_dict["loss/raw_grid_reg_loss"] = opt.lambda_trans_reg * (grid_trans_reg_loss.detach() if isinstance(grid_trans_reg_loss, torch.Tensor) else float(grid_trans_reg_loss)),
                    log_dict["loss/raw_density_reg_loss"] = opt.lambda_density_reg * (density_reg.detach() if isinstance(density_reg, torch.Tensor) else float(density_reg)),
            
        else: # if opt.lambda_rgb_loss is 0, skip loss calculation
            loss = 0
            Ll1 = 0
            if pipe.use_wandb:
                log_dict = {
                    "loss/step": iteration,
                }
        
        # error handling
        if (loss.isnan() + loss.isinf()) > 0:
            for k, v in log_dict.items():
                print(f"log_dict: {k} | {v}")
            raise ValueError("Loss is NaN unexpected behavior")
            
        # =========================  update gaussians w/ reconstruction loss  =========================
        
        if loss > 0:
            loss.backward()
            # densify actor gaussians
            if iteration > dataset.iter_smpl_densify and (iteration < dataset.iter_densify_smpl_until): # default: 3500
                with torch.no_grad():
                    densify_prune_people_infos(
                                            fname=fname, 
                                            people_infos=people_infos, 
                                            opt=human_train_opt, 
                                            scene_extent=scene.cameras_extent, 
                                            d_opacity=d_opacity,
                                            visibility_filter=visibility_filter, 
                                            radii=radii, 
                                            viewspace_point_tensor=viewspace_point_tensor,
                                            scene=scene, 
                                            scene_gaussians=scene_gaussians,
                                            pipe=pipe,
                                            )
            # prune actor gaussians after densify
            elif iteration > dataset.iter_smpl_densify and (iteration < dataset.iter_prune_smpl_until): # default: 7000
                with torch.no_grad():
                    prune_points_people_infos(
                                            fname=fname, 
                                            people_infos=people_infos, 
                                            opt=human_train_opt, 
                                            scene_extent=scene.cameras_extent,
                                            d_opacity=d_opacity
                                            )
            
            # update refine network from recon loss
            if dataset.use_deform:
                torch.nn.utils.clip_grad_norm_(deform.deform.parameters(), max_norm=1.0)
                deform.optimizer.step()
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

            # update people
            for person_info in people_infos:
                if fname not in person_info.fnames or not person_info.detected_bbox[int(fname.split('_')[-1]) - 1]:
                    continue

                # update gaussian parameters
                person_info.gaussians.optimizer.step()
                person_info.gaussians.optimizer.zero_grad(set_to_none = True)
                person_info.misc['optimized_step'] += 1

                # update optimizers
                if person_info.do_trans_grid:
                    person_info.grid_optimizer.step()
                    person_info.grid_optimizer.zero_grad()
                    person_info.smpl_deformer.last_trans = []

                if do_smpl_fitting_local:
                    person_info.local_pose_optimizer.step()
                    person_info.local_pose_optimizer.zero_grad()

                if do_smpl_fitting_global:
                    person_info.global_pose_optimizer.step()
                    person_info.global_pose_optimizer.zero_grad()
                    
                # prune infnan points
                lbs_grid_offset = None
                lbs_grid_scale = None
                if hasattr(person_info.smpl_deformer , "lbs_voxel_final"):
                    lbs_grid_offset = person_info.smpl_deformer.offset.clone().reshape(1, -1)
                    lbs_grid_scale = person_info.smpl_deformer.scale.clone().reshape(1, -1)                
                n_nan = person_info.gaussians.prune_infnan_points(offset=lbs_grid_offset, scale=lbs_grid_scale)
                if n_nan > 0:
                    if not f"_{int(person_info.person_number):03}/n_nan" in log_dict:
                        log_dict[f"_{int(person_info.person_number):03}/n_nan"] = n_nan
                    else:
                        log_dict[f"_{int(person_info.person_number):03}/n_nan"] += n_nan
            
            loss = loss.detach()    

        # get new residuals after densify or prune
        if iteration < dataset.start_deform or not dataset.use_deform:
            d_color, d_opacity = 0.0, 0.0
        else:
            d_color, d_opacity = get_residuals(people_infos, deform, fname, total_frame)
        
        # ==============================  calculate SDS loss  ==============================

        if dataset.use_diffusion_guidance and (iteration % dataset.apply_dgm_every_n == 0) and (iteration > dataset.dgm_start_iter):
            density_reg_loss_weight = 0 if not dataset.use_density_reg_loss else opt.lambda_density_reg
            density_reg_loss_weight = density_reg_loss_weight * lambda_rgb  
            dg_losses, dglog_dict = diffusion_renderer(
                                                        DGM, 
                                                        viewpoint_cam, 
                                                        scene_gaussians, 
                                                        people_infos, 
                                                        pipe, 
                                                        background, 
                                                        deform, # update refine network with diffusion guidance loss
                                                        d_color, d_opacity,
                                                        scaling_modifier=1.0, 
                                                        override_color=None, 
                                                        iteration=iteration, 
                                                        do_optim=True, 
                                                        dgm_loss_weight=opt.lambda_dg_loss, 
                                                        cd_loss_weight=opt.lambda_cd_loss, 
                                                        non_directional_visibility=dataset.dgm_hard_masking,
                                                        num_inference_steps=dataset.dgm_num_inference_steps,
                                                        minimum_mask_thrs=dataset.dgm_minimum_mask_thrs,
                                                        masking_optimizer=dataset.dgm_use_optimizer_masking,
                                                        cfg_rescale_weight=dataset.dgm_cfg_rescale_weight,
                                                        density_reg_loss_weight=density_reg_loss_weight,
                                                        grid_trans_reg_loss_weight=opt.lambda_trans_reg * lambda_rgb  
                                                        )
            loss += dg_losses.detach().squeeze() if isinstance(dg_losses, torch.Tensor) else dg_losses      # Just for logging

            # append guidance loss to log_dict
            if pipe.use_wandb:
                log_dict["loss/loss_w_guide"] = loss.detach()
                
                dg_losses = 0
                cd_losses = 0
                for pnum, dg_loss in dglog_dict.items():
                    log_dict[f"dg_loss/{pnum}_dgloss"] = dg_loss['dg_loss'].detach()
                    log_dict[f"_{pnum}/dg_iter"] = DGM.step[pnum]
                    log_dict[f"_{pnum}/dg_lambda"] = DGM.dg_lambda[pnum] * opt.lambda_dg_loss

                    dg_losses += dg_loss['dg_loss'].detach().cpu()
                    if 'cd_loss' in dg_loss:
                        log_dict[f"_{pnum}/cd_lambda"] = DGM.cd_lambda[pnum] * opt.lambda_cd_loss
                        cd_losses += dg_loss['cd_loss'].detach().cpu()

                    for k, v in dg_loss.items():
                        if f"_{pnum}/{k}" in log_dict:
                            log_dict[f"_{pnum}/{k}"] += v
                        else:
                            log_dict[f"_{pnum}/{k}"] = v
                log_dict["loss/controlnet_guide"] = dg_losses
                log_dict["loss/color-consistency"] = cd_losses
        
        # append total loss
        log_dict["loss/tot_loss"] = loss.detach() if isinstance(loss, torch.Tensor) else loss
                
        iter_end.record()
        
        # ==============================  save files  ==============================
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * (loss.item() if isinstance(loss, torch.Tensor) else loss) + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                n_gauss = scene_gaussians.get_n_points
                for pi in people_infos:
                    n_gauss += pi.gaussians.get_n_points
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}, N_gauss: {n_gauss}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if (iteration in save_iterations):
                print(f"[INFO] Saving data at iteration {iteration}")
                save_path = Path(scene.model_path) / f"iteration_{iteration}"
                save_path.mkdir(parents=True, exist_ok=True)
                
                # save stage point cloud
                scene.gaussians.save_ply(save_path / "stage_pcd.ply")
                
                for person_info in people_infos:
                    # save actor point cloud
                    person_info.gaussians.save_ply(save_path / f"actor_{int(person_info.person_number):03}.ply")

                    # save SMPL parameters
                    smpl_params_tensor = torch.cat([
                        person_info.smpl_scale.reshape(1,1).repeat(len(person_info.smpl_global_poses), 1),
                        person_info.smpl_global_poses,
                        person_info.smpl_local_poses
                    ], dim=-1)
                    smpl_params = smpl_params_tensor.detach().cpu()
                    pandas.to_pickle(smpl_params, save_path / f"smpl_params_{int(person_info.person_number):03}.pkl")

                if dataset.use_deform:
                    deform.save_weights(save_path)

            # Do rendering
            if (iteration in test_iterations):
                print(f"[INFO] Rendering results at iteration {iteration}")
                os.makedirs(scene.model_path + "/training", exist_ok=True)
                render_full_video(Path(scene.model_path)/ "training", scene, people_infos, 
                                  pipe, background, deform, dataset.start_deform, dataset.use_deform, iteration, type="gt", render_only_people=False)
                render_full_video(Path(scene.model_path)/ "training", scene, people_infos, 
                                  pipe, background, deform, dataset.start_deform, dataset.use_deform, iteration, type=pipe.render_type) # circle / wave / fix
                render_full_canonical(Path(scene.model_path) / "training", people_infos, pipe, white_bg, d_color, d_opacity, iteration)

            if iteration % pipe.log_interval == 0:
                image_log_path = Path(scene.model_path)/ "training" / "log_images"
                image_log_path.mkdir(parents=True, exist_ok=True)
                render_log_images(image_log_path, viewpoint_cam, scene_gaussians, people_infos, pipe, background, d_color, d_opacity, iteration)
                
                canonical_log_path = Path(scene.model_path)/ "training" / "log_canonical"
                canonical_log_path.mkdir(parents=True, exist_ok=True)
                render_canonical_images(canonical_log_path, people_infos, pipe, white_bg, d_color, d_opacity, iteration)

                if dataset.use_deform:
                    deform_image_path = Path(scene.model_path)/ "training" / "deform_images"
                    deform_image_path.mkdir(parents=True, exist_ok=True)
                    render_deform_images(deform_image_path, viewpoint_cam, scene_gaussians, people_infos, pipe, background, d_color, d_opacity, iteration)

        # ==============================  additional operations  ==============================
        
        with torch.no_grad():
            # Clip SHs
            if dataset.iter_clip_person_shs > 0:
                if iteration % dataset.iter_clip_person_shs:
                    for person_info in people_infos:
                        n_clipped = person_info.gaussians.clip_invalid_shs()
                        if pipe.use_wandb:
                            pnum = person_info.person_number
                            log_dict[f"_{pnum}/n_clipped"] = n_clipped

            # fix SMPL initial vertices
            if iteration < dataset.iter_fix_smpl_init_verts:
                for person_info in people_infos:
                    person_info.gaussians.fix_smpl_init_position()       # reset gaussians to have initial points
        
        # ==============================  logging  ==============================
        # Append additional informations in logger
        if pipe.use_wandb:
            for pi in people_infos:
                pnum = int(pi.person_number)
                p_n_gaussian = pi.gaussians.get_n_points
                person_iteration = pi.misc['optimized_step']
                log_dict[f"_{pnum}/n_gaussian"] = p_n_gaussian
                log_dict[f"_{pnum}/step"] = person_iteration
                
                lbs_grid_offset = None
                lbs_grid_scale = None
                if hasattr(pi.smpl_deformer , "lbs_voxel_final"):
                    lbs_grid_offset = pi.smpl_deformer.offset.clone().reshape(1, -1)
                    lbs_grid_scale = pi.smpl_deformer.scale.clone().reshape(1, -1)
                n_nan = pi.gaussians.prune_infnan_points(offset=lbs_grid_offset, scale=lbs_grid_scale)
                
                if n_nan > 0:
                    if not f"_{pnum}/n_nan" in log_dict:
                        log_dict[f"_{pnum}/n_nan"] = n_nan
                    else:
                        log_dict[f"_{pnum}/n_nan"] += n_nan

                valid_idx = pi.gaussians.denom > 0
                if (valid_idx.sum() > 0) and iteration % 10 == 0:
                    grads = torch.norm(pi.gaussians.xyz_gradient_accum[valid_idx] / pi.gaussians.denom[valid_idx], dim=-1).squeeze()
                    log_dict[f"_{pnum}/grads_max"] = grads.max()
                    log_dict[f"_{pnum}/grads_min"] = grads.min()
                    log_dict[f"_{pnum}/grads_min"] = grads.std()
                    log_dict[f"_{pnum}/n_valid"] = (valid_idx.sum())
                    log_dict[f"_{pnum}/valid_ratio"] = (valid_idx.sum()) / p_n_gaussian
                    log_dict[f"_{pnum}/over_thrs"] = (grads >= human_train_opt.densify_grad_threshold).sum()
                    
        # Do logging
        if pipe.use_wandb:
            wandb.log(log_dict)

###########################################################################################
##                             Person densification functions                            ##
###########################################################################################

def densify_prune_people_infos(fname, people_infos: List, opt, scene_extent, d_opacity, visibility_filter, radii, viewspace_point_tensor, scene, scene_gaussians, pipe):
    gaussian_n_offset = 0
    for p_id, person_info in enumerate(people_infos):
        if fname not in person_info.fnames or not person_info.detected_bbox[int(fname.split('_')[-1]) - 1]:
            continue
        _gaussians = person_info.gaussians
        person_visibility_filter = visibility_filter[gaussian_n_offset:gaussian_n_offset+_gaussians.get_n_points]
        person_radii = radii[gaussian_n_offset:gaussian_n_offset+_gaussians.get_n_points]
        person_viewspace_point_tensor_grad = viewspace_point_tensor[p_id].grad          # [_idx:_idx+pi.gaussians.get_n_points]
        _gaussians.max_radii2D[person_visibility_filter] = torch.max(_gaussians.max_radii2D[person_visibility_filter], person_radii[person_visibility_filter])
        _gaussians.add_densification_stats(None, person_visibility_filter, person_viewspace_point_tensor_grad)
        
        person_iteration = person_info.misc['optimized_step']

        if isinstance(d_opacity, torch.Tensor):
            _d_opacity = d_opacity[gaussian_n_offset:gaussian_n_offset+_gaussians.get_n_points]
        else:
            _d_opacity = d_opacity
        
        gaussian_n_offset += person_info.gaussians.get_n_points

        if person_iteration > opt.densify_from_iter and person_iteration % opt.densification_interval == 0: # and person_iteration < opt.densify_until_iter
            size_threshold = 20
            _gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, 1, size_threshold, \
                d_opacity=_d_opacity, delete_large=True)


def prune_points_people_infos(fname, people_infos, opt, scene_extent, d_opacity):
    gaussian_n_offset = 0
    for person_info in people_infos:
        if fname not in person_info.fnames or not person_info.detected_bbox[int(fname.split('_')[-1]) - 1]:
            continue

        _gaussians = person_info.gaussians

        person_iteration = person_info.misc['optimized_step']
        
        if isinstance(d_opacity, torch.Tensor):
            _d_opacity = d_opacity[gaussian_n_offset:gaussian_n_offset+_gaussians.get_n_points]
        else:
            _d_opacity = d_opacity
        
        gaussian_n_offset += _gaussians.get_n_points
        
        if person_iteration > opt.densify_from_iter and person_iteration % opt.densification_interval == 0: # and person_iteration < opt.densify_until_iter 
            size_threshold = None
            extent = 5 * person_info.smpl_scale.item()
            person_info.gaussians.prune_gaussians(min_opacity=0.005, d_opacity=_d_opacity, \
                extent=extent, max_screen_size=size_threshold, min_scale=None)    

###########################################################################################
##                                    Main function                                      ##
###########################################################################################

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # basic options
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10, 3_000, 7_000, 10_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10, 3_000, 7_000, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--exp_name", type=str, default ="debug")
    # additional options
    parser.add_argument('--bg_radius', type=float, default=10.)
    parser.add_argument("--use_bg_reg", action='store_true')
    parser.add_argument("--reverse_mask", action='store_true')
    parser.add_argument("--use_deform", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("============================================================")
    print(f"\n[INFO] Optimizing {Path(args.model_path).name}. Experiment name: {args.exp_name}\n")
    print("============================================================")
    
    if args.use_deform:
        print("[INFO] Using refinement network")
    
    lp_extracted = lp.extract(args)
    lp_extracted.bg_radius = args.bg_radius
    lp_extracted.use_bg_reg = args.use_bg_reg
    lp_extracted.reverse_mask = args.reverse_mask
    lp_extracted.use_deform = args.use_deform

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp_extracted, 
        op.extract(args), 
        pp.extract(args), 
        args.save_iterations, 
        args.test_iterations, 
        args.debug_from, 
        args.exp_name
    )

    # All done
    print("\nTraining complete.")
