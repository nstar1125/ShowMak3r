import tyro
import viser

import torch
import tyro
import open3d as o3d
import numpy as np
import time
import cv2
import viser
import viser.transforms as tf
import matplotlib
import random
from pathlib import Path
from typing import List
from scipy.spatial.transform import Rotation as R

from showmak3r.pipeline.scene.gaussian_model import GaussianModel
from showmak3r.pipeline.refine.deform_branch import get_residuals

from showmak3r.utils.camera_utils import focal2fov
from showmak3r.utils.general_utils import rot_weighting, build_rotation, unstrip_symmetric
from showmak3r.utils.sh_utils import eval_sh
from showmak3r.pipeline.renderer.renderer_wrapper import calculate_residuals_full

SINGLE_ONLY = False
RENDER_INTERVAL = 3 # non-single-only mode
TARGET_FID = 20 # single-only mode

def get_actor_splat(viewpoint_camera, 
                    people_infos: List,
                    d_color, d_opacity, #residuals
                    scaling_modifier = 1.0, 
                    override_color = None, 
                    hard_rendering=False, 
                    render_normal=False):
    
    uid = viewpoint_camera.uid

    means3D_people = []
    colors_precomp_people = []
    opacities_people = []
    cov3D_precomp_people = []
    
    residual_idx_offset = 0 # residual offset for deformation output
    
    # ------------------------- Deform actor gaussians -------------------------
    for person_idx, person_info in enumerate(people_infos):
        person_pc = person_info.gaussians

        if (not person_info.detected_bbox[uid]):
            continue
        
        beta = person_info.beta

        # load smpl_param
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

        means3D = person_pc.get_xyz # (N, 3)
        opacity = person_pc.get_opacity # (N, 1)

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
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
            shs_view = person_pc.get_features.transpose(1, 2).view(-1, 3, (person_pc.max_sh_degree+1)**2) # (N, 3, 9)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(person_pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # Rotate in canonical space 
            dir_pp_normalized = torch.einsum('bij,bj->bi', smpl_rots.transpose(1,2), dir_pp_normalized)
            dir_pp_normalized = dir_pp_normalized/dir_pp_normalized.norm(dim=1, keepdim=True)

            sh2rgb = eval_sh(person_pc.active_sh_degree, shs_view, dir_pp_normalized) # deg, sh, dirs
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if len(override_color) != len(means3D):
                colors_precomp = override_color.reshape(-1, 3).repeat(len(means3D), 1)
            else:
                colors_precomp = override_color
                
        if hard_rendering:
            opacity[opacity > 0.1] /= opacity[opacity > 0.1].detach()
            opacity[opacity <= 0.1] *= 0
        
        num_gaussians = means3D.shape[0]
        if isinstance(d_color, torch.Tensor) and isinstance(d_opacity, torch.Tensor):
            person_d_color = d_color[residual_idx_offset:residual_idx_offset+num_gaussians]
            person_d_opacity = d_opacity[residual_idx_offset:residual_idx_offset+num_gaussians]
        else:
            person_d_color = d_color
            person_d_opacity = d_opacity
        
        residual_idx_offset+=num_gaussians    

        means3D_people.append(means3D)
        colors_precomp_people.append(torch.clamp(colors_precomp+person_d_color, min=0.0, max=1.0))
        opacities_people.append(torch.clamp(opacity+person_d_opacity, min=0.0, max=1.0))
        cov3D_precomp_people.append(unstrip_symmetric(cov3D_precomp))
    
    # ------------------------- Concatenate actor gaussians -------------------------
    try:
        # concate the tensors
        if means3D is not None:
            means3D = torch.cat(means3D_people, dim=0).contiguous()
        else:
            means3D = None
            
        if colors_precomp is not None:
            colors_precomp = torch.cat(colors_precomp_people, dim=0).contiguous()
        else:
            colors_precomp = None
        
        if opacity is not None:
            opacity = torch.cat(opacities_people, dim=0).contiguous()
        else:
            opacity = None
        
        if cov3D_precomp is not None:
            cov3D_precomp = torch.cat(cov3D_precomp_people, dim=0).contiguous()
        else:
            cov3D_precomp = None
        
    except NameError:
        return None
    
    # ------------------------- Change to viser format -------------------------
    
    centers = means3D.detach().cpu().numpy()
    rgbs = colors_precomp.detach().cpu().numpy()
    opacity = opacity.detach().cpu().numpy()
    covariances = cov3D_precomp.detach().cpu().numpy()
    
    return {
        "centers": centers,
        "rgbs": rgbs,
        "opacities": opacity,
        "covariances": covariances,
    }

def get_stage_splat(pc : GaussianModel):
    centers = pc.get_xyz.detach().cpu().numpy()
    opacity = pc.get_opacity.detach().cpu().numpy()
    
    scales = pc.get_scaling
    rotations = pc.get_rotation

    R_mats = build_rotation(rotations)
    covariances = R_mats @ torch.diag_embed(scales**2) @ R_mats.transpose(-2, -1)
    covariances = covariances.detach().cpu().numpy()

    # SH -> RGB
    feats = pc.get_features                                      # (N, C, (D+1)^2)
    shs_view = feats.transpose(1, 2).reshape(-1, 3, (pc.max_sh_degree+1)**2)
    sh2rgb = eval_sh(0, shs_view, None) # SH level 0
    colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
    rgbs = colors.detach().cpu().numpy()
    
    return {
        "centers": centers,
        "rgbs": rgbs,
        "opacities": opacity,
        "covariances": covariances,
    }

def visualize_4dgs(
    scene,
    people_infos: List, 
    pipe,
    background,
    deform_pipe,
    share: bool = True,
    port: int = 8080,
):
    train_cameras = scene.getTrainCameras().copy()
    test_cameras = scene.getTestCameras().copy()
    if len(test_cameras) > 0:
        full_cameras = train_cameras + test_cameras
    else:
        full_cameras = train_cameras
    full_cameras = sorted(full_cameras.copy(), key=lambda x: x.fname)
    
    if deform_pipe is not None:
        print("Calculating mean opacity and color for interpolation")
        res_dict = calculate_residuals_full(scene, people_infos, pipe, background, deform_pipe, full_cameras)
        mean_opacity = res_dict["mean_opacity"]
        mean_color = res_dict["mean_color"]
        gmask_list = res_dict["gmask_list"]
        op_list = res_dict["op_list"]
        col_list = res_dict["col_list"]
    
    num_frames = len(full_cameras)
    if not SINGLE_ONLY:
        num_frames = num_frames // RENDER_INTERVAL

    ######################################################################################
    ##                                  Set viser server                                ##
    ###################################################################################### 
    server = viser.ViserServer(port=port)
    if share:
        server.request_share_url()
    
    server.scene.set_up_direction('-y')
    if not SINGLE_ONLY:
        with server.gui.add_folder("Playback"):
            gui_timestep = server.gui.add_slider(
                "Timestep",
                min=0,
                max=num_frames - 1,
                step=1,
                initial_value=0,
                disabled=False,
            )
            gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
            gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
            gui_playing = server.gui.add_checkbox("Playing", True)
            gui_view_all_frames = server.gui.add_checkbox("View All Frames", False)
            gui_framerate = server.gui.add_slider(
                "FPS", min=1, max=60, step=1, initial_value=10
            )
        
        # gui settings
        @gui_next_frame.on_click
        def _(_):
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        @gui_prev_frame.on_click
        def _(_):
            gui_timestep.value = (gui_timestep.value - 1) % num_frames

        @gui_playing.on_update
        def _(_):
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

        def update_frame_visibility():
            frame_nodes[gui_timestep.value].visible = True
            time.sleep(0.2)
            for i, frame_node in enumerate(frame_nodes):
                if i == gui_timestep.value:
                    continue
                if not gui_view_all_frames.value:
                    frame_node.visible = False
                else:
                    frame_node.visible = True

        prev_timestep = 0
        @gui_timestep.on_update
        def _(_):
            nonlocal prev_timestep
            current_timestep = gui_timestep.value
            update_frame_visibility()
            prev_timestep = current_timestep        

        @gui_view_all_frames.on_update
        def _(_):
            update_frame_visibility()
    
    ######################################################################################
    ##                                  Upload assets                                   ##
    ######################################################################################
    # add stage gaussians
    scene_gaussians = scene.gaussians
    stage_splat = get_stage_splat(scene_gaussians)
    server.scene.add_gaussian_splats(
        f"/stage",
        centers=stage_splat["centers"],
        rgbs=stage_splat["rgbs"],
        opacities=stage_splat["opacities"],
        covariances=stage_splat["covariances"],
    )
    if SINGLE_ONLY:
        # add actor gaussians
        d_color, d_opacity = get_residuals(people_infos, deform_pipe, full_cameras[TARGET_FID].fname, len(full_cameras))
        actor_splat = get_actor_splat(
            full_cameras[TARGET_FID],
            people_infos,
            d_color=d_color, d_opacity=d_opacity,
        )
        server.scene.add_gaussian_splats(
            f"/actors",
            centers=actor_splat["centers"],
            rgbs=actor_splat["rgbs"],
            opacities=actor_splat["opacities"],
            covariances=actor_splat["covariances"],
        )
    
    frame_nodes = []
    for i, viewpoint_cam in enumerate(full_cameras):
        if i % RENDER_INTERVAL != 0 or SINGLE_ONLY:
            continue
        
        frame_node = server.scene.add_frame(f"/frames/t_{i}", show_axes=False)
        frame_nodes.append(frame_node)

        # add actor gaussians
        if deform_pipe is not None:
            d_opacity = op_list[i] * gmask_list[i] + mean_opacity * (~gmask_list[i]) # (N, 1)
            d_color = col_list[i] * gmask_list[i] + mean_color * (~gmask_list[i]) # (N, 3)
        else:
            d_color, d_opacity = 0.0, 0.0
        
        actor_splat = get_actor_splat(
            viewpoint_cam,
            people_infos,
            d_color=d_color, d_opacity=d_opacity,
        )
        server.scene.add_gaussian_splats(
            f"/frames/t_{i}",
            centers=actor_splat["centers"],
            rgbs=actor_splat["rgbs"],
            opacities=actor_splat["opacities"],
            covariances=actor_splat["covariances"],
        )

    ######################################################################################
    ##                                  Start visualization                             ##
    ######################################################################################
    def playback_loop():
        nonlocal prev_timestep
        while True:
            if not SINGLE_ONLY and gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1) % num_frames
                time.sleep(1.0 / gui_framerate.value)
            else:
                time.sleep(10.0)

    print("[INFO] Start viser visualization.")
    playback_loop()