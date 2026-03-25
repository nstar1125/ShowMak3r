import torch
import numpy as np
import math
import random
from typing import Optional, List
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


from showmak3r.utils.sh_utils import eval_sh
from showmak3r.utils.general_utils import rot_weighting, build_rotation, unstrip_symmetric
from showmak3r.utils.graphics_utils import project_points_to_cam
from showmak3r.utils.jnts_utils import extract_square_bbox, filter_invisible_joints
from showmak3r.utils.image_utils import tensor2cv, get_crop_img
from showmak3r.utils.loss_utils import get_cd_loss, denisty_reg_loss
from showmak3r.utils.draw_op_jnts import smpl_joints2op_joints, draw_op_img

from showmak3r.pipeline.smpl_deform.deformer import SMPLDeformer
from showmak3r.pipeline.scene.gaussian_model import GaussianModel
from showmak3r.pipeline.diffusion.guidance.joint_utils import filter_invisible_face_joints_w_prompts, get_view_prompt_of_body, combine_prompts

@torch.no_grad()
def render_visibility_mask(view_camera, smpl_deformer, raster_settings, person_gaussian, smpl_param, means3D, means2D, opacity, scales, rotations, cov3D_precomp, is_hard_rendering=True, non_directional=False, visibility_mask=None, masking_optimizer=False):
    if visibility_mask is None:
        # get view-direction
        dir_pp = (means3D.detach() - view_camera.camera_center)
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)

        # convert it to back to canonical space
        R = smpl_deformer.get_rotations(person_gaussian.get_xyz, smpl_param, cond=None)     # (N, 3, 3)
        R = R.permute(0, 2, 1)      # inverse the rotation

        # Need to convert back to original space
        canon_dir_pp = torch.einsum('bij,bj->bi', R, dir_pp_normalized)

        # Get weighted mask
        # 1: visible, 0: invisible, 0~1 partially visible.
        visibility_mask = person_gaussian.get_viz_mask(canon_dir_pp, non_directional_vizmask=non_directional)                # Currently, we are using cosine as visibility weight
        
    if masking_optimizer:
        return None, visibility_mask.reshape(-1, 1)

    
    # set color as black for visible points (especially, the inner product)
    colors_precomp = torch.ones_like(means3D) * visibility_mask.reshape(-1,1)   # (N. 3)    


    # Set color as white_bg
    white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") # for testing
    raster_settings = raster_settings._replace(bg=white_bg)

        
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if is_hard_rendering:
        opacity[opacity > 0.2] = 1
        opacity[opacity <= 0.2] = 0

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    ras_outputs = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    rendered_image = ras_outputs[0]
    mask = rendered_image.squeeze().mean(0)

    return mask


def diffusion_renderer(
    DGM, 
    viewpoint_camera, 
    scene_pc : GaussianModel, 
    people_infos: List, 
    pipe, 
    bg_color : torch.Tensor, 
    deform,
    d_color, d_opacity,
    scaling_modifier = 1.0, 
    override_color = None, 
    hard_rendering=False, 
    offsets=None, 
    offset_id=0, 
    iteration: int=-1,
    render_normal=False, 
    do_optim: bool=False, 
    dgm_loss_weight: float=0.1, 
    cd_loss_weight: float=1,
    non_directional_visibility: bool=False,
    num_inference_steps: int=10,
    minimum_mask_thrs: float=0.02,
    masking_optimizer: bool=False,
    cfg_rescale_weight: float=0.8,
    density_reg_loss_weight: float=0.0,
    grid_trans_reg_loss_weight: float=0.0,
    only_density_reg_loss: bool=False,
    ):
    
    fname = viewpoint_camera.fname
    # ==============================  diffusion guidance loss per actor  ==============================
    log_dict = dict()
    losses = torch.tensor(0., device='cuda')
    
    gaussian_visibility_list = []
    screenspace_points_list = []
    radii_list = []
    gs_idx_offset = 0
    for person_idx, person_info in enumerate(people_infos):
        person_pc = person_info.gaussians

        # skip if not in train_pnums
        if not (person_info.person_number in DGM.train_pnums):
            continue
        
        if fname not in person_info.fnames:
            print("Person not in frame! Optimize anyway.")
            _data_idx = random.randint(0, len(person_info.fnames)-1) 
            person_info.misc['optimized_step'] += 1
        # elif data_idx == -1:
        #     _data_idx = random.randint(0, len(person_info.fnames)-1)
        else:
            _data_idx = person_info.fnames.index(fname)
        
        beta = person_info.beta

        if offsets is not None and person_idx+1 == offset_id:
            p_offset = torch.tensor(offsets, device=person_info.smpl_global_poses.device)
        else:
            p_offset = torch.tensor([0.0, 0.0, 0.0], device=person_info.smpl_global_poses.device)
        
        # load smpl_param
        smpl_param = torch.cat([
            person_info.smpl_scale.reshape(-1),
            person_info.smpl_global_poses[_data_idx] + p_offset,
            person_info.smpl_local_poses[_data_idx],
            beta
        ], dim=-1)
        smpl_param = smpl_param.unsqueeze(0)
        uid = person_info.uids[_data_idx]

        smpl_deformer = person_info.smpl_deformer
        xyz = person_pc.get_xyz

        screenspace_points = torch.zeros_like(xyz, dtype=person_pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        means3D = xyz
        means2D = screenspace_points
        opacity = person_pc.get_opacity
        screenspace_points_list.append(screenspace_points) # used later
        
        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = person_pc.get_covariance(scaling_modifier)
        
        if getattr(person_info, 'view_dir_reg', False):
            _rotations = build_rotation(person_pc.get_rotation)
        else:
            _rotations = None
            
        # transform points according to SMPL
        cond = dict(
            img_idx=uid
        )
        # set smpl_param for SDS loss
        smpl_param[0, 0] = 1.       # Fix scale as 1
        smpl_param[0, 1:4] *= 0     # remove global translation
        smpl_param[0, 2] = 0.3      # remove global translation (transl + 0.3 on y direction)
        smpl_param[0, 4:7] *= 0     # remove global rotation
        smpl_scale = smpl_param[0, 0]

        # Now select camera from camera sampler
        pnum = person_info.person_number
        mini_cam, invert_bg_color, mini_cam_vers, mini_cam_hors, mini_cam_radii, smpl_param, camera_type_name = DGM.get_render_camera(
                                        pnum, 
                                        smpl_scale.item(), 
                                        get_single_fixed_camera=False,
                                        smpl_param = smpl_param,
                                        smpl_deformer=smpl_deformer,
                                        )
        
        # transform points according to SMPL
        means3D, cov3D_precomp, _rotations = smpl_deformer.deform_gp(means3D, cov3D_precomp, smpl_param, cond=cond, rotations=_rotations)
        
        if getattr(person_info, 'view_dir_reg', False):
            dir_pp = (means3D.detach() - mini_cam.camera_center.repeat(person_pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            
            occ_weights = rot_weighting(_rotations, dir_pp_normalized)
            opacity = opacity * occ_weights

        # Part for normal-rendering
        if render_normal and getattr(person_info, 'view_dir_reg', False):
            _, rot_vectors = rot_weighting(_rotations, dir_pp_normalized, return_rot_vector=True)
            override_color = rot_vectors / 2 + 0.5

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            # safe 
            if pipe.convert_SHs_python or True:
                shs_view = person_pc.get_features.transpose(1, 2).view(-1, 3, (person_pc.max_sh_degree+1)**2)
                dir_pp = (means3D - mini_cam.camera_center.repeat(person_pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(person_pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = person_pc.get_features
        else:
            colors_precomp = override_color
            

        if hard_rendering:
            opacity = torch.ones_like(opacity).float().to(opacity.device)

        num_gaussians = means3D.shape[0]
        if isinstance(d_color, torch.Tensor) and isinstance(d_opacity, torch.Tensor):
            person_d_color = d_color[gs_idx_offset:gs_idx_offset+num_gaussians]
            person_d_opacity = d_opacity[gs_idx_offset:gs_idx_offset+num_gaussians]
        else:
            person_d_color = d_color
            person_d_opacity = d_opacity
        
        gs_idx_offset+=num_gaussians    

        # Get rendering
        tanfovx = math.tan(mini_cam.FoVx * 0.5)
        tanfovy = math.tan(mini_cam.FoVy * 0.5)
        render_bg_color = 1 - bg_color if invert_bg_color else bg_color

        raster_settings = GaussianRasterizationSettings(
            image_height=int(mini_cam.image_height),
            image_width=int(mini_cam.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=render_bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=mini_cam.world_view_transform,
            projmatrix=mini_cam.full_proj_transform,
            sh_degree=scene_pc.active_sh_degree,
            campos=mini_cam.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        ras_outputs = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = torch.clamp(colors_precomp+person_d_color, min=0.0, max=1.0),
            opacities = torch.clamp(opacity+person_d_opacity, min=0.0, max=1.0),
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        rendered_image = ras_outputs[0]
        radii = ras_outputs[1]
        radii_list.append(radii)
        rendered_image = rendered_image.unsqueeze(0)

        # ----------------------------  calculate diffusion guidance loss  ----------------------------
        
        # clip the highest value
        rendered_image[rendered_image > 1] /= rendered_image[rendered_image > 1]

        if density_reg_loss_weight > 0:
            assert (len(ras_outputs) > 2), "Invalid rasterizer (don't return alpha)"
            alpha = ras_outputs[-1]
            density_reg_loss = denisty_reg_loss(alpha) * density_reg_loss_weight
        else:
            density_reg_loss = 0

        grid_trans_reg_loss = 0
        if person_info.do_trans_grid:
            grid_trans_reg_loss += (person_info.smpl_deformer.last_trans[-1] ** 2).mean()
            grid_trans_reg_loss = grid_trans_reg_loss * grid_trans_reg_loss_weight

        if only_density_reg_loss:
            # When skipping DGM loss calculating
            log_dict[person_info.person_number] = dict()
            log_dict[person_info.person_number]['density_reg_loss'] = (density_reg_loss.detach().cpu() if isinstance(density_reg_loss, torch.Tensor) else 0.0)
            log_dict[person_info.person_number]['trans_reg_loss_novel_view'] = (grid_trans_reg_loss.detach().cpu() if isinstance(grid_trans_reg_loss, torch.Tensor) else 0.0)
            _loss = density_reg_loss + grid_trans_reg_loss
        else:
            if masking_optimizer:
                print(f"[INFO] we mask optimizers instead of rendering visibility")
                assert do_optim, f"[ERROR] masking optimizer is only valid with OPT_INDIV options"
                mask, gaussian_visibility = render_visibility_mask(
                                mini_cam, 
                                smpl_deformer, 
                                raster_settings, 
                                person_pc, 
                                smpl_param, means3D, means2D, opacity, scales, rotations, cov3D_precomp,
                                non_directional=non_directional_visibility,
                                masking_optimizer=masking_optimizer)
                
                gaussian_visibility += minimum_mask_thrs
                gaussian_visibility = torch.clamp(gaussian_visibility, 0, 1)
                gaussian_visibility_list.append(gaussian_visibility)
            else:
                mask = None

            dgm_cond = []
            new_prompt = None

            if DGM.enable_controlnet:
                with torch.no_grad():
                    smpl_output = person_info.smpl_deformer.smpl_server(smpl_param)
                    smpl_jnts = smpl_output['smpl_jnts'].detach().cpu()
                    image_res = (512, 512)
                    pj_jnts = project_points_to_cam(mini_cam, smpl_jnts.squeeze().numpy(), image_res=image_res)
                    op_joints = smpl_joints2op_joints(pj_jnts)
                    op_3d_jnt = smpl_joints2op_joints(smpl_jnts.squeeze().numpy())
                    
                    # get prompts
                    lower_body_prompt = get_view_prompt_of_body(op_3d_jnt, mini_cam, is_lower_body=True)
                    upper_body_prompt = get_view_prompt_of_body(op_3d_jnt, mini_cam, is_lower_body=False)
                    filtered_op_3d_jnts, head_prompt = filter_invisible_face_joints_w_prompts(op_3d_jnt, mini_cam, image_res=image_res)
                    new_prompt = combine_prompts(head_prompt, upper_body_prompt, lower_body_prompt, op_joints, image_res)

                    # # filter with visibility
                    # op_joints = filter_invisible_joints(op_joints)
                    for idx, _op_3d_jnt in enumerate(filtered_op_3d_jnts):
                        if _op_3d_jnt is None:
                            op_joints[idx] = None
                
                    op_cond_img = draw_op_img(op_joints, 512)
                    dgm_cond.append(op_cond_img)
                        
            if cd_loss_weight == 0:
                # Turn of cd_loss
                DGM.color_correction = False
            else:
                DGM.color_correction = True
                
            # get diffusion guidance loss
            dg_loss, step_ratio, guid_loss_dict = DGM.get_loss(
                                                rendered_image, 
                                                pnum, 
                                                vers=mini_cam_vers, 
                                                hors=mini_cam_hors, 
                                                radii=mini_cam_radii, 
                                                iteration=iteration, 
                                                cond_image=dgm_cond, 
                                                additional_prompt=new_prompt,
                                                mask=mask,
                                                num_inference_steps=num_inference_steps,
                                                save_intermediate=True,
                                                img_description=camera_type_name,
                                                minimum_mask_thrs=minimum_mask_thrs,
                                                cfg_rescale_weight=cfg_rescale_weight
                                                )
            _loss = dg_loss * dgm_loss_weight + density_reg_loss + grid_trans_reg_loss # dg_loss scale: initially, around 129

            # error handling
            if (dg_loss.isnan().sum() + dg_loss.isinf().sum()) > 0:
                raise ValueError("[ERROR] Diffusion Guidance Loss is NaN or Inf")

            # logging
            log_dict[person_info.person_number] = dict()
            log_dict[person_info.person_number]['dg_loss'] = _loss.detach().cpu()
            log_dict[person_info.person_number]['density_reg_loss'] = (density_reg_loss.detach().cpu() if isinstance(density_reg_loss, torch.Tensor) else 0.0)
            log_dict[person_info.person_number]['trans_reg_loss_novel_view'] = (grid_trans_reg_loss.detach().cpu() if isinstance(grid_trans_reg_loss, torch.Tensor) else 0.0)
            log_dict[person_info.person_number]['noise_ratio'] = step_ratio
            log_dict[person_info.person_number]['render_height'] = int(mini_cam.image_height)

            if len(guid_loss_dict) > 0:
                for k, v in guid_loss_dict.items():
                    log_dict[person_info.person_number][k] = v

        # Color consistency Loss
        if cd_loss_weight > 0 and False:
            lambda_cd = DGM.get_lambda_cd(pnum)
            lambda_cd = lambda_cd * cd_loss_weight
            if lambda_cd == 0:
                continue

            # get GT pixel sets
            gt_pixel_lists = person_info.misc['color_distribution']
            
            rendered_image = rendered_image.squeeze()
            
            rendered_pixel_lists = rendered_image.reshape(3, -1).T
            rendered_pixel_lists = rendered_pixel_lists[rendered_pixel_lists.sum(-1) > 0]   # remove black bg
            rendered_pixel_lists = rendered_pixel_lists[rendered_pixel_lists.sum(-1) < 3]   # remove white  bg
            
            cd_loss = get_cd_loss(gt_pixel_lists, rendered_pixel_lists)
            cd_loss = cd_loss * cd_loss_weight
            log_dict[person_info.person_number]['cd_loss'] = cd_loss.detach().cpu()
            _loss += cd_loss

        losses += _loss
        
    # ==============================  backward pass  ==============================
    if do_optim:
        losses.backward()

        # update deformation
        if isinstance(d_color, torch.Tensor) and isinstance(d_opacity, torch.Tensor):
            torch.nn.utils.clip_grad_norm_(deform.deform.parameters(), max_norm=1.0)
            deform.optimizer.step()
            deform.optimizer.zero_grad()

        # update gaussians per actor
        for person_idx, person_info in enumerate(people_infos):
            person_pc = person_info.gaussians
        
            if masking_optimizer:
                # remove gradient with mask
                # thrs = (step_ratio / DGM.guidance_controlnet.num_train_timesteps) 
                thrs = 0.5          # naive on off      
                mask = gaussian_visibility_list[person_idx] < thrs           # higher thrs -> large change -> (less optimized points) (here visibility is highest if it's 0)
                gaussian_viz_mask = mask.squeeze()
                person_info.gaussians.prune_gradients(gaussian_viz_mask)
                print("Do pruning ")
                
            # Here, DGM.opt is equal to trainer.opt -> so it's fine to use same variable name here
            if person_info.misc['optimized_step'] >= DGM.opt.density_start_iter and person_info.misc['optimized_step'] <= DGM.opt.density_end_iter:
                viewspace_point_tensor, visibility_filter, radii = screenspace_points_list[person_idx], (radii_list[person_idx] > 0), radii_list[person_idx]
                
                # if masking_optimizer:
                #     visibility_filter[gaussian_viz_mask] = False
                    
                person_info.gaussians.max_radii2D[visibility_filter] = torch.max(person_info.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                person_info.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            
            if (person_info.misc['optimized_step'] > DGM.opt.iter_smpl_densify) and (person_info.misc['optimized_step'] <= DGM.opt.iter_prune_smpl_until):
                if person_info.misc['optimized_step'] % DGM.opt.densification_interval == 0:
                    person_info.gaussians.prune_gaussians(0.005, 5, None)
                
            # set optimizer as zero
            person_info.gaussians.optimizer.step()
            person_info.gaussians.optimizer.zero_grad(set_to_none = True)

            if person_info.do_trans_grid:
                person_info.grid_optimizer.step()
                person_info.grid_optimizer.zero_grad()
                person_info.smpl_deformer.last_trans = []
                
                
            n_nan = person_info.gaussians.prune_infnan_points()
            if n_nan > 0:
                log_dict[person_info.person_number]['n_nan'] = n_nan

    losses = losses.detach()

    return losses, log_dict
