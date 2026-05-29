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

import torch
import numpy as np
import math
import random
from typing import Optional, List
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


from showmak3r.utils.sh_utils import eval_sh
from showmak3r.utils.general_utils import rot_weighting, build_rotation, unstrip_symmetric
from showmak3r.utils.graphics_utils import project_points_to_cam, fov2focal
from showmak3r.utils.jnts_utils import extract_square_bbox, filter_invisible_joints
from showmak3r.utils.image_utils import tensor2cv, get_crop_img
from showmak3r.utils.loss_utils import get_cd_loss, denisty_reg_loss
from showmak3r.utils.draw_op_jnts import smpl_joints2op_joints, draw_op_img

from showmak3r.pipeline.smpl_deform.deformer import SMPLDeformer
from showmak3r.pipeline.scene.gaussian_model import GaussianModel
from showmak3r.pipeline.diffusion.guidance.joint_utils import filter_invisible_face_joints_w_prompts, get_view_prompt_of_body, combine_prompts


def stage_render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, opacity_thrs=-1, render_normal: bool=False, get_viz_mask: bool=False):
    """
    Render the stage. 
    Background tensor (bg_color) must be on GPU.
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation


    # Part for view-dir regularization
    if pipe.view_dir_reg:
        rotations = pc.get_rotation
        dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        
        occ_weights = rot_weighting(build_rotation(rotations), dir_pp_normalized)
        opacity = opacity * occ_weights

    
    # Part for normal-rendering
    if render_normal and pipe.view_dir_reg:
        _, rot_vectors = rot_weighting(build_rotation(rotations), dir_pp_normalized, return_rot_vector=True)
        override_color = rot_vectors / 2 + 0.5


    # Part of rendering visible mask
    if get_viz_mask and pipe.view_dir_reg:
        bg_color = torch.zeros_like(bg_color)
        override_color = torch.ones_like(means3D)
        opacity[occ_weights>0] = 1 

        # To set black on invisible parts
        override_color[occ_weights==0] = 0


    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    if opacity_thrs > 0:
        opacity[opacity < opacity_thrs] = opacity[opacity < opacity_thrs] * 0
        
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    ras_outputs = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    rendered_image = ras_outputs[0]
    radii = ras_outputs[1]
    depth = ras_outputs[2]
    alpha = ras_outputs[3]
    
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "depth": depth,
        "alpha": alpha
    }


def composite_render(viewpoint_camera, 
                      scene_pc : GaussianModel, 
                      people_infos: List, pipe, 
                      bg_color : torch.Tensor, 
                      d_color, d_opacity, #residuals
                      scaling_modifier = 1.0, 
                      override_color = None, 
                      hard_rendering=False, 
                      render_only_people=False, 
                      offsets=None, offset_id=0, 
                      render_normal=False, 
                      get_deformed_points=False,
                      delete_pid=None,
                      insert_id=0
                      ):
    """
    Render the stage and actors. 
    """
    uid = viewpoint_camera.uid

    means3D_people = []
    means2D_people = []
    shs_people = []
    colors_precomp_people = []
    opacities_people = []
    scales_people = []
    rotations_people = []
    cov3D_precomp_people = []
    screenspace_points_people = []
    
    residual_idx_offset = 0 # residual offset for deformation output
    
    # ------------------------- Deform actor gaussians -------------------------
    for person_idx, person_info in enumerate(people_infos):
        if delete_pid == person_idx:
            continue
        
        person_pc = person_info.gaussians

        if (not person_info.detected_bbox[uid]):
            continue
        
        fname = person_info.fnames[uid] 
        beta = person_info.beta

        # load smpl_param
        if offsets is not None and person_idx+1 == offset_id:
            p_offset = torch.tensor(offsets, device=person_info.smpl_global_poses.device)
        else:
            p_offset = torch.tensor([0.0, 0.0, 0.0], device=person_info.smpl_global_poses.device)

        # load smpl_param
        smpl_param = torch.cat([
            person_info.smpl_scale.reshape(-1),
            person_info.smpl_global_poses[uid] + p_offset,
            person_info.smpl_local_poses[uid],
            beta
        ], dim=-1)
        smpl_param = smpl_param.unsqueeze(0)
        uid = person_info.uids[uid]
        
        smpl_deformer = person_info.smpl_deformer

        xyz = person_pc.get_xyz
        
        screenspace_points = torch.zeros_like(xyz, dtype=person_pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        means3D = xyz # (N, 3)
        means2D = screenspace_points
        opacity = person_pc.get_opacity # (N, 1)

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
        means3D, cov3D_precomp, smpl_rots = smpl_deformer.deform_gp(means3D, cov3D_precomp, smpl_param, cond=cond, rotations=_rotations)
        
        if _rotations is not None and smpl_rots is not None:
            _rotations = torch.bmm(smpl_rots, _rotations)
    
        if getattr(person_info, 'view_dir_reg', False):
            dir_pp = (means3D.detach() - viewpoint_camera.camera_center.repeat(person_pc.get_features.shape[0], 1))
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
        colors_precomp = None # RGB (N, 3) 
        if override_color is None: # run
            # safe 
            if pipe.convert_SHs_python or True:
                shs_view = person_pc.get_features.transpose(1, 2).view(-1, 3, (person_pc.max_sh_degree+1)**2) # (N, 3, 9)
                dir_pp = (means3D - viewpoint_camera.camera_center.repeat(person_pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                # Rotate in canonical space 
                dir_pp_normalized = torch.einsum('bij,bj->bi', smpl_rots.transpose(1,2), dir_pp_normalized)
                dir_pp_normalized = dir_pp_normalized/dir_pp_normalized.norm(dim=1, keepdim=True)

                sh2rgb = eval_sh(person_pc.active_sh_degree, shs_view, dir_pp_normalized) # deg, sh, dirs
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = person_pc.get_features
        else:
            if len(override_color) != len(means3D):
                colors_precomp = override_color.reshape(-1, 3).repeat(len(means3D), 1)
            else:
                colors_precomp = override_color
                
        if hard_rendering:
            opacity[opacity > 0.1] /= opacity[opacity > 0.1].detach()
            opacity[opacity <= 0.1] *= 0
            # opacity = torch.ones_like(opacity).float().to(opacity.device)
        
        num_gaussians = means3D.shape[0]
        if isinstance(d_color, torch.Tensor) and isinstance(d_opacity, torch.Tensor):
            person_d_color = d_color[residual_idx_offset:residual_idx_offset+num_gaussians]
            person_d_opacity = d_opacity[residual_idx_offset:residual_idx_offset+num_gaussians]
        else:
            person_d_color = d_color
            person_d_opacity = d_opacity
        
        if insert_id == person_idx+1: # insert only models without deform
            person_d_color = 0.
            person_d_opacity = 0.
        
        residual_idx_offset+=num_gaussians    

        means3D_people.append(means3D)
        means2D_people.append(means2D)
        shs_people.append(shs)
        colors_precomp_people.append(torch.clamp(colors_precomp+person_d_color, min=0.0, max=1.0))
        opacities_people.append(torch.clamp(opacity+person_d_opacity, min=0.0, max=1.0))
        scales_people.append(scales)
        rotations_people.append(rotations)
        cov3D_precomp_people.append(cov3D_precomp)
        screenspace_points_people.append(screenspace_points)
        
    # ------------------------- Add stage gaussians -------------------------
    if render_only_people:
        pass
    else:
        xyz = scene_pc.get_xyz
            
        screenspace_points = torch.zeros_like(xyz, dtype=scene_pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
    
        means3D = xyz
        means2D = screenspace_points
        opacity = scene_pc.get_opacity 

        if render_only_people:
            opacity = opacity * 0

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        
        cov3D_precomp = scene_pc.get_covariance(scaling_modifier)

        if pipe.view_dir_reg:
            dir_pp = (xyz - viewpoint_camera.camera_center.repeat(scene_pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            
            occ_weights = rot_weighting(build_rotation(scene_pc.get_rotation), dir_pp_normalized)
            opacity = opacity * occ_weights
        # Part for normal-rendering
        if render_normal and pipe.view_dir_reg:
            _, rot_vectors = rot_weighting(build_rotation(scene_pc.get_rotation), dir_pp_normalized, return_rot_vector=True)
            override_color = rot_vectors / 2 + 0.5

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python or True:
                shs_view = scene_pc.get_features.transpose(1, 2).view(-1, 3, (scene_pc.max_sh_degree+1)**2)
                dir_pp = (xyz - viewpoint_camera.camera_center.repeat(scene_pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(scene_pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = scene_pc.get_features
        else:
            if render_normal and not pipe.view_dir_reg:
                colors_precomp = bg_color.clone().reshape(-1,3).repeat(len(means3D), 1)
            elif len(override_color.reshape(-1)) == 3:
                colors_precomp = override_color.reshape(-1, 3).repeat(len(means3D), 1)
            else:
                colors_precomp = override_color
            
        means3D_people.append(means3D)
        means2D_people.append(means2D)
        shs_people.append(shs)
        colors_precomp_people.append(colors_precomp)
        opacities_people.append(opacity)
        scales_people.append(scales)
        rotations_people.append(rotations)
        cov3D_precomp_people.append(cov3D_precomp)
        screenspace_points_people.append(screenspace_points)
    
    # ------------------------- Concatenate stage and actor gaussians -------------------------
    try:
        # concate the tensors
        if means3D is not None:
            means3D = torch.cat(means3D_people, dim=0).contiguous()
        else:
            means3D = None

        if means2D is not None:
            means2D = torch.cat(means2D_people, dim=0).contiguous()
        else:
            means2D = None
            
        if shs is not None and shs_people[0] is not None:
            shs = torch.cat(shs_people, dim=0).contiguous()
        else:
            shs = None
            
        if colors_precomp is not None:
            colors_precomp = torch.cat(colors_precomp_people, dim=0).contiguous()
        else:
            colors_precomp = None
        
        if opacity is not None:
            opacity = torch.cat(opacities_people, dim=0).contiguous()
        else:
            opacity = None
            
        if scales is not None:
            scales = torch.cat(scales_people, dim=0).contiguous()
        else:
            scales = None
            
        if rotations is not None:
            rotations = torch.cat(rotations_people, dim=0).contiguous()
        else:
            rotations = None
                
        if cov3D_precomp is not None:
            cov3D_precomp = torch.cat(cov3D_precomp_people, dim=0).contiguous()
        else:
            cov3D_precomp = None
        
    except NameError:
        return None

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=scene_pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    # ------------------------- Rasterize Gaussians -------------------------

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    ras_outputs = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    rendered_image = ras_outputs[0]
    radii = ras_outputs[1]
    depth = ras_outputs[2]
    alpha = ras_outputs[3]        

    _return = {"render": rendered_image,
            "viewspace_points": screenspace_points_people,        # it's same as screen_space_points
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth,
            "mask": alpha
            }
    
    # if viewpoint_camera.uid == 0:
    #     import open3d as o3d
    #     print(f"Export PCD of frame_{viewpoint_camera.fname} for visualization.")
    #     scene_pcd = o3d.geometry.PointCloud()
    #     opacity_map = (opacity > 0.0)
    #     opacity_map[scene_pc.get_xyz.shape[0]:, :] = True # keep scene
    #     colors_precomp[:-scene_pc.get_xyz.shape[0], :] = torch.tensor([0.5, 1.0, 0.5])
    #     scene_pcd.points = o3d.utility.Vector3dVector((means3D*opacity_map.expand(-1, 3)).cpu().numpy())
    #     scene_pcd.colors = o3d.utility.Vector3dVector((colors_precomp*opacity_map.expand(-1, 3)).cpu().numpy())
    #     output_path = "/workspace/ShowMak3r_RELEASE/frame_pcd.ply"
        
    #     o3d.io.write_point_cloud(output_path, scene_pcd)
    
    # calculate mean3D in uv space
    ones = torch.ones((means3D.shape[0], 1), device=means3D.device, dtype=means3D.dtype)
    Xw_h = torch.cat([means3D, ones], dim=1)
    
    Xc_h = Xw_h @ viewpoint_camera.world_view_transform
    Xc   = Xc_h[:, :3]
    x, y, z = Xc[:, 0], Xc[:, 1], Xc[:, 2]
    fx = fov2focal(viewpoint_camera.FoVx, viewpoint_camera.image_width)
    fy = fov2focal(viewpoint_camera.FoVy, viewpoint_camera.image_height)
    _u = fx * (x / z) + viewpoint_camera.image_width // 2
    _v = fy * (y / z) + viewpoint_camera.image_height // 2
    means3D_uv = torch.stack([_v, _u], dim=-1).round().long()
    
    _return['uv'] = means3D_uv.detach()
    
    if get_deformed_points:
        _return['means3D'] = means3D.detach()
    
    return _return

def canonical_render(viewpoint_camera, 
                     pc : GaussianModel, 
                     pipe, 
                     bg_color : torch.Tensor, 
                     d_color, d_opacity, 
                     scaling_modifier = 1.0, 
                     override_color = None, 
                     deformer: Optional[SMPLDeformer]=None, 
                     smpl_param=None, 
                     get_viz_mask: bool=False, 
                     deformer_cond=None):
    """
    Render canonical smpls. 
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    if get_viz_mask and pipe.view_dir_reg:
        bg_color = torch.ones_like(bg_color)    # Should allow inpainting of BG here

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python or True:
        ### WE MUST CALCULATE HERE!!!
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        
    if pipe.view_dir_reg:
        _rotations = build_rotation(pc.get_rotation)
    else:
        _rotations = None
        
    # transform points according to SMPL
    smpl_rots = None
    if not deformer is None:
        if smpl_param is None:
            smpl_param = viewpoint_camera.smpl_param
            if smpl_param is not None:
                smpl_param = smpl_param.to(means3D.device).float()
        cond = dict(
            img_idx=viewpoint_camera.uid
        )
        if deformer_cond is not None:
            for k, v in deformer_cond.items():
                if k not in cond:
                    cond[k] = v

        means3D, cov3D_precomp, smpl_rots = deformer.deform_gp(means3D, cov3D_precomp, smpl_param, cond=cond)
        if _rotations is not None and smpl_rots is not None:
            _rotations = torch.bmm(smpl_rots, _rotations)
            
    
    # Part of view-direction regularizer
    if pipe.view_dir_reg:
        dir_pp = (means3D.detach() - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        
         
        occ_weights = rot_weighting(_rotations, dir_pp_normalized)
        
        if not get_viz_mask:
            opacity = opacity * occ_weights
        

    # Part of rendering visible mask
    if get_viz_mask and pipe.view_dir_reg:
        override_color = torch.ones_like(means3D)
        override_color[occ_weights.squeeze()>0] *= 0 

        # To set black on invisible parts
        # override_color[occ_weights.squeeze()==0] *= 0
        # opacity[occ_weights.squeeze()==0] = torch.ones_like(opacity[occ_weights.squeeze()==0])

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python or True:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            if smpl_rots is not None:
                # Rotate in canonical space 
                dir_pp_normalized = torch.einsum('bij,bj->bi', smpl_rots.transpose(1,2), dir_pp_normalized)
                dir_pp_normalized = dir_pp_normalized/dir_pp_normalized.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    ras_outputs = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp+d_color,
        opacities = opacity+d_opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    rendered_image = ras_outputs[0]
    radii = ras_outputs[1]
    
    _return = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

    return _return