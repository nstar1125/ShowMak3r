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


import os
import json
import random
from os import makedirs
from argparse import ArgumentParser
from typing import List, Union, NamedTuple, Optional, Dict
from pathlib import Path
import copy

import cv2
import pandas
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# from showmak3r.pipeline.scene import Scene, HumanScene
from showmak3r.pipeline.scene.gaussian_model import GaussianModel
from showmak3r.pipeline.renderer.gaussian_renderer import stage_render, composite_render, canonical_render
from showmak3r.pipeline.renderer.smpl_renderer import render_smpl
from showmak3r.pipeline.refine.deform_branch import get_residuals # deformation model
from showmak3r.pipeline.dataset.composite_loader import PersonTrain
from showmak3r.pipeline.dataset.camera_loader import gen_canon_cams
from showmak3r.pipeline.scene.cameras import Camera

from showmak3r.utils.io_utils import save4images, save2images, save_rgb_image, save_video
from showmak3r.utils.image_utils import psnr, draw_bbox_w_pid, draw_bodypose_with_color
from showmak3r.utils.system_utils import searchForMaxIteration


# ============================== for training stage ==============================
def render_stage_full(scene, gaussians, pipe, background, iteration, result_path):
    '''
    Save video of the all rendered views of the stage.
    '''
    viewpoint_list = scene.getTrainCameras().copy()
    save_path = result_path / f"iteration_{iteration}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    for idx, viewpoint_cam in tqdm(enumerate(viewpoint_list), desc="Rendering", total=len(viewpoint_list)):
        render_pkg = stage_render(viewpoint_cam, gaussians, pipe, background)
        render_image, viewspace_point_tensor, visibility_filter, radii, render_depth = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]
        render_image[render_image > 1] /= render_image[render_image > 1] # (turn off saturated points)
        gt_image = viewpoint_cam.gt_image
        gt_depth = viewpoint_cam.pseudo_gt_depth
        
        render_depth_norm = (render_depth - render_depth.min()) / (render_depth.max() - render_depth.min())
        gt_depth_norm = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min())
        render_depth_image = render_depth_norm.repeat(3, 1, 1) 
        gt_depth_image = gt_depth_norm.repeat(3, 1, 1)
        
        render_path = save_path / f"{idx:05d}.png"
        save4images(render_image, gt_image, render_depth_image, gt_depth_image, path=render_path)
    save_video(save_path, f"iteration_{iteration}.mp4", fps=30, format="png")

def render_depth_maps(scene, gaussians, pipe, background, save_path):
    '''
    Save depth maps of the all rendered views of the stage.
    '''
    viewpoint_list = scene.getTrainCameras().copy()
    for idx, viewpoint_cam in tqdm(enumerate(viewpoint_list), desc="Rendering depth", total=len(viewpoint_list)):
        render_pkg = stage_render(viewpoint_cam, gaussians, pipe, background)
        render_image, viewspace_point_tensor, visibility_filter, radii, render_depth = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]
        render_depth = render_depth.detach().cpu().numpy()
        render_depth = render_depth[0, :, :]
        np.save(save_path / f"{viewpoint_cam.fname}.npy", render_depth)

# ============================== for training position ==============================
def render_smpl_full(
    cfg,
    save_path,
    video_name,
    img_dict,
    camdicts,
    shot_dicts,
    smpl_model,
    device
):
    '''
    Save video with the projected SMPLs.
    '''
    smpl_render_dicts, smpl_render_alpha_dicts = render_smpl(
        camdicts,
        shot_dicts,
        smpl_model,
        device
    )
    for fname, smpl_render_img in tqdm(smpl_render_dicts.items(), total=len(smpl_render_dicts), desc="Exporting"):
        fid = int(fname.split("_")[-1])-1
        alpha_rendering = smpl_render_alpha_dicts[fname] / 255.
        alpha_rendering = alpha_rendering[..., None]
        img = img_dict[fname]
        alpha_rendering *= 0.8
        img = img * (1 - alpha_rendering) + smpl_render_img * alpha_rendering
        img = img.copy()
        for shot_id, shot_dict in shot_dicts.items():
            for pnum, person_dict in shot_dict.items():
                if fname not in person_dict:
                    continue
                frame_result = person_dict[fname]
                if frame_result['bbox'] is not None:
                    img = draw_bbox_w_pid(img, frame_result['bbox'], (255, 0, 0), pid=pnum)
                if frame_result['body'] is not None:
                    _body_jnts = []
                    for jnt in frame_result['body']:
                        if jnt.isnan().all():
                            _body_jnts.append(None)
                        elif jnt[-1] < cfg.joint_threshold:
                            _body_jnts.append(None)
                        else:
                            _body_jnts.append(jnt)
                    img = draw_bodypose_with_color(img, _body_jnts, img.shape[:2], (0, 255, 0), mode='op25')
        cv2.imwrite(str(save_path / f"{fid:05d}.jpg"), img)
    save_video(save_path, f"{video_name}.mp4", fps=30, format="jpg")

# ============================== for training actors ==============================
def render_log_images(save_path, viewpoint_cam, scene_gaussians, people_infos, pipe, background, d_color, d_opacity, iteration):
    '''
    Save log image of the GT and rendered actors and the stage.
    '''
    log_actor_render_pkg = composite_render(viewpoint_cam, scene_gaussians, people_infos, pipe, background, d_color, d_opacity, 
                                            scaling_modifier = 1.0, override_color = None, render_only_people=True)
    log_comp_render_pkg = composite_render(viewpoint_cam, scene_gaussians, people_infos, pipe, background, d_color, d_opacity, 
                                            scaling_modifier = 1.0, override_color = None, render_only_people=False)
    log_stage_render_pkg = stage_render(viewpoint_cam, scene_gaussians, pipe, background)
    gt_image = viewpoint_cam.gt_image
    
    log_img_wo_stage = log_actor_render_pkg["render"]
    log_img_wo_stage[log_img_wo_stage > 1] /= log_img_wo_stage[log_img_wo_stage > 1] # (turn off saturated points)
    log_img_w_stage = log_comp_render_pkg["render"]
    log_img_w_stage[log_img_w_stage > 1] /= log_img_w_stage[log_img_w_stage > 1] # (turn off saturated points)
    log_img_stage = log_stage_render_pkg["render"]
    log_img_stage[log_img_stage > 1] /= log_img_stage[log_img_stage > 1] # (turn off saturated points)
    save4images(log_img_w_stage, log_img_wo_stage, log_img_stage, gt_image, path=save_path / f"{int(iteration):05d}.png")

def render_deform_images(save_path, viewpoint_cam, scene_gaussians, people_infos, pipe, background, d_color, d_opacity, iteration):
    '''
    Save ablation image of the refinement.
    '''
    comp_render_pkg = composite_render(viewpoint_cam, scene_gaussians, people_infos, pipe, background, 0., 0., 
                                            scaling_modifier = 1.0, override_color = None, render_only_people=False)
    comp_render_w_deform_pkg = composite_render(viewpoint_cam, scene_gaussians, people_infos, pipe, background, d_color, d_opacity, 
                                            scaling_modifier = 1.0, override_color = None, render_only_people=False)

    comp_img_w_deform = comp_render_w_deform_pkg["render"]
    comp_img_w_deform[comp_img_w_deform > 1] /= comp_img_w_deform[comp_img_w_deform > 1] # (turn off saturated points)
    
    comp_img_wo_deform = comp_render_pkg["render"]
    comp_img_wo_deform[comp_img_wo_deform > 1] /= comp_img_wo_deform[comp_img_wo_deform > 1] # (turn off saturated points)
    
    save2images(comp_img_wo_deform, comp_img_w_deform, path=save_path / f"{int(iteration):05d}.png")

def render_canonical_images(save_path, people_infos, pipe, background, d_color, d_opacity, iteration):
    '''
    Save canonical images of the actors.
    '''
    canon_views = gen_canon_cams(res=512)
    cam_back = canon_views[0]
    cam_front = canon_views[len(canon_views)//2]

    residual_idx_offset = 0 # residual offset for deformation output
    for idx, person_info in enumerate(people_infos):
        pnum = person_info.person_number
        
        person_save_path = save_path / str(pnum)
        person_save_path.mkdir(exist_ok=True)
        
        person_info = people_infos[idx]

        num_gaussians = person_info.gaussians.get_xyz.shape[0]
        if isinstance(d_color, torch.Tensor) and isinstance(d_opacity, torch.Tensor):
            person_d_color = d_color[residual_idx_offset:residual_idx_offset+num_gaussians]
            person_d_opacity = d_opacity[residual_idx_offset:residual_idx_offset+num_gaussians]
        else:
            person_d_color = d_color
            person_d_opacity = d_opacity

        # Render front & back 
        front_canon_image = canonical_render(cam_front, person_info.gaussians, pipe, background, person_d_color, person_d_opacity)["render"].detach().cpu()
        back_canon_image = canonical_render(cam_back, person_info.gaussians, pipe, background, person_d_color, person_d_opacity, deformer=person_info.smpl_deformer)["render"].detach().cpu()
        front_canon_image[front_canon_image > 1] /= front_canon_image[front_canon_image > 1] # (turn off saturated points)
        back_canon_image[back_canon_image > 1] /= back_canon_image[back_canon_image > 1] # (turn off saturated points)

        save2images(front_canon_image, back_canon_image, path=person_save_path / f"{iteration:05d}.png")
        residual_idx_offset+=num_gaussians

def render_full_video(
    save_path,
    scene,
    people_infos,
    pipe,
    background,
    deform_pipe, start_deform, use_deform,
    iteration, 
    type="gt",
    render_only_people=False,
    delete_pid=None,
    offset_pid=None,
    offset_xyz=None,
):
    '''
    Save video of the all rendered views of the scene with various camera trajectories.
    '''
    train_cameras = scene.getTrainCameras().copy()
    test_cameras = scene.getTestCameras().copy()
    if len(test_cameras) > 0:
        full_cameras = train_cameras + test_cameras
    else:
        full_cameras = train_cameras
    full_cameras = sorted(full_cameras.copy(), key=lambda x: x.fname)
    scene_gaussians = scene.gaussians
    
    # calculate mean opacity and color
    if iteration >= start_deform and use_deform:
        print("Calculating mean opacity and color for interpolation")
        res_dict = calculate_residuals_full(scene, people_infos, pipe, background, deform_pipe, full_cameras, delete_pid)
        mean_opacity = res_dict["mean_opacity"]
        mean_color = res_dict["mean_color"]
        gmask_list = res_dict["gmask_list"]
        op_list = res_dict["op_list"]
        col_list = res_dict["col_list"]

    if type == "mean" or type == "mean_circle":
        Rs = np.array([cam.R for cam in full_cameras])
        Ts = np.array([cam.T for cam in full_cameras])
        
        # Smooth Rs using slerp and Ts using gaussian filter
        from scipy.ndimage import gaussian_filter1d
        from scipy.spatial.transform import Rotation as R
        sigma = 10.0
        
        # Convert rotation matrices to quaternions for slerp
        rotations = R.from_matrix(Rs)
        quats = rotations.as_quat()
        
        # Apply gaussian smoothing to quaternions
        smooth_quats = np.zeros_like(quats)
        for i in range(quats.shape[1]):
            smooth_quats[:, i] = gaussian_filter1d(quats[:, i], sigma=sigma, mode='nearest')
        
        # Normalize quaternions and convert back to rotation matrices
        smooth_quats = smooth_quats / np.linalg.norm(smooth_quats, axis=1, keepdims=True)
        smooth_Rs = R.from_quat(smooth_quats).as_matrix()
        
        # Smooth translation vectors
        smooth_Ts = np.zeros_like(Ts)
        for i in range(Ts.shape[1]):
            smooth_Ts[:, i] = gaussian_filter1d(Ts[:, i], sigma=sigma, mode='nearest')
    
    # render
    for cam_id, viewpoint_cam in tqdm(enumerate(full_cameras), desc="Rendering", total=len(full_cameras)):
        if iteration < start_deform or not use_deform:
            d_color, d_opacity = 0.0, 0.0
        else:
            d_opacity = op_list[cam_id] * gmask_list[cam_id] + mean_opacity * (~gmask_list[cam_id]) # (N, 1)
            d_color = col_list[cam_id] * gmask_list[cam_id] + mean_color * (~gmask_list[cam_id]) # (N, 3)
        
        if type == "gt":
            render_path = save_path / f"render_{int(iteration):05d}"
            render_path.mkdir(parents=True, exist_ok=True)
            cam = viewpoint_cam
        elif type == "wave" or type == "circle":
            render_path = save_path / f"novel_{int(iteration):05d}"
            render_path.mkdir(parents=True, exist_ok=True)
            
            if type == "wave": # zoom in and out
                interval = 100
                scale = 12.0
                offset = (0., 0., -2) # -0.8
                shift = (0, 0, -np.sin(cam_id * (2 * np.pi / interval)) * scale)
            elif type == "circle": # circle around the trajectory
                interval = 100
                scale = 1 # 0.045 
                offset = (0, 0, 0) # (0.0, 0.0, -0.09)
                shift = (np.sin(cam_id * (2 * np.pi / interval)) * scale, np.cos(cam_id * (2 * np.pi / interval)) * scale, 0)
            cam = Camera(uid=viewpoint_cam.uid,
                  fname=viewpoint_cam.fname,
                  R=viewpoint_cam.R,
                  T=np.array([viewpoint_cam.T[0]+shift[0]+offset[0], viewpoint_cam.T[1]+shift[1]+offset[1], viewpoint_cam.T[2]+shift[2]+offset[2]]),
                  FoVx=viewpoint_cam.FoVx,
                  FoVy=viewpoint_cam.FoVy,
                  cx=viewpoint_cam.cx,
                  cy=viewpoint_cam.cy,
                  image = None, mask = None, depth = None,
                  image_width=viewpoint_cam.image_width,
                  image_height=viewpoint_cam.image_height)
        elif type == "fix" or type == "fix_circle":
            render_path = save_path / f"novel_{int(iteration):05d}"
            render_path.mkdir(parents=True, exist_ok=True)
            
            fix_idx = 0
            if type == "fix":
                offset = (0.0, 0, -0.2)
                shift = (0, 0, 0)
            elif type == "fix_circle":
                interval = 100
                # scale = 0.2
                scale = 12
                offset = (0, 0, 0) # (0,0,0.1)
                shift = (np.sin(cam_id * (2 * np.pi / interval)) * scale * 0.7, 0, np.cos(cam_id * (2 * np.pi / interval)) * scale * 1.2)
                # shift = (np.sin(cam_id * (2 * np.pi / interval)) * scale * 0.7, 0, np.cos(cam_id * (2 * np.pi / interval)) * scale * 1.2)
            
            cam = Camera(uid=viewpoint_cam.uid,
                  fname=viewpoint_cam.fname,
                  R=train_cameras[fix_idx].R,
                  T=np.array([train_cameras[fix_idx].T[0]+shift[0]+offset[0], train_cameras[fix_idx].T[1]+shift[1]+offset[1], train_cameras[fix_idx].T[2]+shift[2]+offset[2]]),
                  FoVx=train_cameras[fix_idx].FoVx * 1.1,
                  FoVy=train_cameras[fix_idx].FoVy * 1.1,
                  cx=train_cameras[fix_idx].cx,
                  cy=train_cameras[fix_idx].cy,
                  image = None, mask = None, depth = None,
                  image_width=train_cameras[fix_idx].image_width,
                  image_height=train_cameras[fix_idx].image_height)
        elif type == "mean" or type == "mean_circle":
            render_path = save_path / f"novel_{int(iteration):05d}"
            render_path.mkdir(parents=True, exist_ok=True)
            
            
            
            if type == "mean":
                offset = (0, 0, 0) # (0.1, 0, 0.4)
                shift = (0, 0, 0)
            elif type == "mean_circle":
                interval = 100
                scale = 0.1
                offset = (0.0, 0.05, 0.0)
                shift = (np.sin(cam_id * (2 * np.pi / interval)) * scale, 0, np.cos(cam_id * (2 * np.pi / interval)) * scale)

            cam = Camera(uid=viewpoint_cam.uid,
                  fname=viewpoint_cam.fname,
                  R=smooth_Rs[cam_id],
                  T=smooth_Ts[cam_id] + np.array(shift) + np.array(offset),
                  FoVx=viewpoint_cam.FoVx,
                  FoVy=viewpoint_cam.FoVy,
                  cx=viewpoint_cam.cx,
                  cy=viewpoint_cam.cy,
                  image = None, mask = None, depth = None,
                  image_width=viewpoint_cam.image_width,
                  image_height=viewpoint_cam.image_height)
        else:
            raise ValueError(f"Invalid type: {type}")
        if delete_pid == -1:
            frame_render_pkg = stage_render(cam, scene_gaussians, pipe, background)
        else:
            frame_render_pkg = composite_render(cam, scene_gaussians, people_infos, pipe, background, d_color, d_opacity, 
                                                scaling_modifier = 1.0, override_color = None, render_only_people=render_only_people, 
                                                offsets=offset_xyz, offset_id=offset_pid, delete_pid=delete_pid)

        render_frame = frame_render_pkg["render"]
        render_frame[render_frame > 1] /= render_frame[render_frame > 1] # (turn off saturated points)
        save_rgb_image(render_frame, path=render_path / f"{cam_id:05d}.png")
    if delete_pid is not None:
        video_name = f"deletion_{iteration:05d}.mp4"
    elif offset_pid is not None:
        video_name = f"relocation_{iteration:05d}.mp4"
    elif type == "gt":
        video_name = f"render_{iteration:05d}.mp4"
    else:
        video_name = f"novel_{iteration:05d}.mp4"
    save_video(render_path, video_name, fps=30)

def render_full_canonical(save_path,
                          people_infos, 
                          pipe, 
                          background,
                          d_color, d_opacity, 
                          iteration):
    '''
    Save video of canonical actors with rotation.
    '''
    canon_views = gen_canon_cams(res=512)

    residual_idx_offset = 0 # residual offset for deformation output
    for idx, person_info in enumerate(people_infos):
        pnum = person_info.person_number
        
        person_save_path = save_path / f"actor_{str(pnum)}"
        person_save_path.mkdir(exist_ok=True)
        
        person_info = people_infos[idx]
            
        for idx, cam in tqdm(enumerate(canon_views), desc="Rendering", total=len(canon_views)):

            num_gaussians = person_info.gaussians.get_xyz.shape[0]
            if isinstance(d_color, torch.Tensor) and isinstance(d_opacity, torch.Tensor):
                person_d_color = d_color[residual_idx_offset:residual_idx_offset+num_gaussians]
                person_d_opacity = d_opacity[residual_idx_offset:residual_idx_offset+num_gaussians]
            else:
                person_d_color = d_color
                person_d_opacity = d_opacity

            # Render front & back 
            canon_image = canonical_render(cam, person_info.gaussians, pipe, background, person_d_color, person_d_opacity)["render"].detach().cpu()
            canon_image[canon_image > 1] /= canon_image[canon_image > 1] # (turn off saturated points)
            save_rgb_image(canon_image, path=person_save_path / f"{idx:05d}.png")
        residual_idx_offset+=num_gaussians
        save_video(person_save_path, f"canonical_{pnum}_{iteration:05d}.mp4", fps=5, format="png")

# ============================== for testing scene ==============================
def render_insertion_video(
    save_path,
    insert_path,
    scene,
    people_infos,
    dataset,
    pipe,
    background,
    deform_pipe, start_deform, use_deform,
    iteration, 
    insert_offset_xyz=None,
):
    '''
    Save video of insertion with a new actor.
    '''
    train_cameras = scene.getTrainCameras().copy()
    test_cameras = scene.getTestCameras().copy()
    if len(test_cameras) > 0:
        full_cameras = train_cameras + test_cameras
    else:
        full_cameras = train_cameras
    full_cameras = sorted(full_cameras.copy(), key=lambda x: x.fname)
    scene_gaussians = scene.gaussians
    
    render_path = save_path / f"insertion_{int(iteration):05d}"
    render_path.mkdir(parents=True, exist_ok=True)

    # calculate mean opacity and color
    if iteration >= start_deform and use_deform:
        print("Calculating mean opacity and color for interpolation")
        res_dict = calculate_residuals_full(scene, people_infos, pipe, background, deform_pipe, full_cameras)
        mean_opacity = res_dict["mean_opacity"]
        mean_color = res_dict["mean_color"]
        gmask_list = res_dict["gmask_list"]
        op_list = res_dict["op_list"]
        col_list = res_dict["col_list"]

    # load insertion gaussian
    loaded_iter = searchForMaxIteration(Path(insert_path))
    smpl_poses = pandas.read_pickle(Path(insert_path) / f"iteration_{loaded_iter}" / f"smpl_params_001.pkl")
    smpl_poses = smpl_poses.cuda()
    smpl_poses[:, 0] *= 1.1
    insert_gaussians = GaussianModel(dataset.sh_degree)
    insert_gaussians.load_ply(os.path.join(insert_path, f"iteration_{loaded_iter}",f"actor_001.ply"), is_insert=True)
    insert_person = PersonTrain(
        uids = people_infos[0].uids,
        fnames = people_infos[0].fnames,
        smpl_local_poses = smpl_poses[:, 4:],
        smpl_scale = smpl_poses[0, 0].squeeze(),
        smpl_global_poses = smpl_poses[:, 1:4], 
        detected_bbox = [True] * len(people_infos[0].fnames),
        local_pose_optimizer = None,
        global_pose_optimizer = None,
        smpl_scale_optimizer = None,
        model_path = insert_path,
        beta = people_infos[0].beta,
        smpl_deformer = people_infos[0].smpl_deformer,
        gaussians = insert_gaussians,
        view_dir_reg = None,
        human_scene = None,
        person_number = str(int(people_infos[-1].person_number)+1),
        do_trans_grid = False,
        trans_grids = None,
        grid_optimizer = None,
        init_smpl_jnts = None,
        cam_centers = None,
        cc_smpl_dir = None,
        representative_img = None,
        misc = dict()
    )
    temp_infos = copy.deepcopy(people_infos)
    temp_infos.append(insert_person)

    for cam_id, viewpoint_cam in tqdm(enumerate(full_cameras), desc="Rendering", total=len(full_cameras)):    
        if cam_id >= smpl_poses.shape[0]:
            print("End of insertion")
            break
        if iteration < start_deform or not use_deform:
            d_color, d_opacity = 0.0, 0.0
        else:
            d_opacity = op_list[cam_id] * gmask_list[cam_id] + mean_opacity * (~gmask_list[cam_id]) # (N, 1)
            d_color = col_list[cam_id] * gmask_list[cam_id] + mean_color * (~gmask_list[cam_id]) # (N, 3)
            # no offset for insertion (TODO - load refine net)
            d_opacity = torch.cat([d_opacity, torch.zeros(insert_person.gaussians.get_n_points, 1, device=d_opacity.device)], dim=0)
            d_color = torch.cat([d_color, torch.zeros(insert_person.gaussians.get_n_points, 3, device=d_color.device)], dim=0)

        frame_render_pkg = composite_render(viewpoint_cam, scene_gaussians, temp_infos, pipe, background, d_color, d_opacity, 
                                                scaling_modifier = 1.0, override_color = None, render_only_people=False, 
                                                offsets=insert_offset_xyz, offset_id=len(temp_infos), insert_id=len(temp_infos))
        render_frame = frame_render_pkg["render"]
        render_frame[render_frame > 1] /= render_frame[render_frame > 1] # (turn off saturated points)
        save_rgb_image(render_frame, path=render_path / f"{cam_id:05d}.png")
    video_name = f"insertion_{iteration:05d}.mp4"
    save_video(render_path, video_name, fps=30)

def render_novel_pose(
    save_path,
    scene,
    pose_data,
    people_infos,
    pipe,
    background,
    deform_pipe, start_deform, use_deform,
    iteration, 
    target_pid=None,
    offset_xyz=None,
):
    '''
    Save video of actors with pose manipulation.
    '''
    train_cameras = scene.getTrainCameras().copy()
    test_cameras = scene.getTestCameras().copy()
    if len(test_cameras) > 0:
        full_cameras = train_cameras + test_cameras
    else:
        full_cameras = train_cameras
    full_cameras = sorted(full_cameras.copy(), key=lambda x: x.fname)
    scene_gaussians = scene.gaussians

    if iteration >= start_deform and use_deform:
        print("Calculating mean opacity and color for interpolation")
        res_dict = calculate_residuals_full(scene, people_infos, pipe, background, deform_pipe, full_cameras)
        mean_opacity = res_dict["mean_opacity"]
        mean_color = res_dict["mean_color"]
        gmask_list = res_dict["gmask_list"]
        op_list = res_dict["op_list"]
        col_list = res_dict["col_list"]

    len_frames = people_infos[target_pid-1].smpl_local_poses.shape[0]
    pose_data[:len_frames, :3] = people_infos[target_pid-1].smpl_local_poses[:, :3]
    mani_person = PersonTrain(
        uids = people_infos[target_pid-1].uids,
        fnames = people_infos[target_pid-1].fnames,
        smpl_local_poses = pose_data[:len_frames].to(people_infos[target_pid-1].smpl_local_poses.device),
        smpl_scale = people_infos[target_pid-1].smpl_scale,
        smpl_global_poses = people_infos[target_pid-1].smpl_global_poses,
        detected_bbox = people_infos[target_pid-1].detected_bbox,
        local_pose_optimizer = None,
        global_pose_optimizer = None,
        smpl_scale_optimizer = None,
        model_path = people_infos[target_pid-1].model_path,
        beta = people_infos[target_pid-1].beta,
        smpl_deformer = people_infos[target_pid-1].smpl_deformer,
        gaussians = people_infos[target_pid-1].gaussians,
        view_dir_reg = None,
        human_scene = None,
        person_number = people_infos[target_pid-1].person_number,
        do_trans_grid = False,
        trans_grids = None,
        grid_optimizer = None,
        init_smpl_jnts = None,
        cam_centers = None,
        cc_smpl_dir = None,
        representative_img = None,
        misc = dict()
    )
    temp_infos = copy.deepcopy(people_infos)
    temp_infos[target_pid-1] = mani_person

    for cam_id, viewpoint_cam in tqdm(enumerate(full_cameras), desc="Rendering", total=len(full_cameras)):
        if iteration < start_deform or not use_deform:
            d_color, d_opacity = 0.0, 0.0   
        else:
            d_opacity = op_list[0] * gmask_list[0] + mean_opacity * (~gmask_list[0]) # (N, 1)
            d_color = col_list[0] * gmask_list[0] + mean_color * (~gmask_list[0]) # (N, 3)
        
        mani_path = save_path / f"manipulate_{int(iteration):05d}"
        mani_path.mkdir(parents=True, exist_ok=True)
        
        frame_render_pkg = composite_render(viewpoint_cam, scene_gaussians, temp_infos, pipe, background, d_color, d_opacity, 
                                            scaling_modifier = 1.0, override_color = None, render_only_people=False, 
                                            offsets=offset_xyz, offset_id=target_pid, delete_pid=-1)

        mani_frame = frame_render_pkg["render"]
        mani_frame[mani_frame > 1] /= mani_frame[mani_frame > 1] # (turn off saturated points)
        save_rgb_image(mani_frame, path=mani_path / f"{cam_id:05d}.png")
    video_name = f"manipulate_{iteration:05d}.mp4"
    save_video(mani_path, video_name, fps=30)

def render_dynamic_frame(
    save_path,
    scene,
    people_infos,
    pipe,
    background,
    deform_pipe, start_deform, use_deform,
    iteration, 
    cam_id = 0 # default is the first camera
):
    '''
    Save video of a single frame with dynamic camera movement.
    '''
    train_cameras = scene.getTrainCameras().copy()
    test_cameras = scene.getTestCameras().copy()
    if len(test_cameras) > 0:
        full_cameras = train_cameras + test_cameras
    else:
        full_cameras = train_cameras
    full_cameras = sorted(full_cameras.copy(), key=lambda x: x.fname)
    scene_gaussians = scene.gaussians

    print("Calculating mean opacity and color for interpolation")
    res_dict = calculate_residuals_full(scene, people_infos, pipe, background, deform_pipe, full_cameras)
    mean_opacity = res_dict["mean_opacity"]
    mean_color = res_dict["mean_color"]
    gmask_list = res_dict["gmask_list"]
    op_list = res_dict["op_list"]
    col_list = res_dict["col_list"]

    viewpoint_cam = full_cameras[cam_id]
    if iteration < start_deform or not use_deform:
        d_color, d_opacity = 0.0, 0.0
    else:
        d_opacity = op_list[cam_id] * gmask_list[cam_id] + mean_opacity * (~gmask_list[cam_id]) # (N, 1)
        d_color = col_list[cam_id] * gmask_list[cam_id] + mean_color * (~gmask_list[cam_id]) # (N, 3)
    
    dynamic_path = save_path / f"dynamic_{int(iteration):05d}"
    dynamic_path.mkdir(parents=True, exist_ok=True)
    
    interval = 100
    for i in tqdm(range(interval), desc="Rendering", total=interval):
        scale = 3.0 # 18 # 0.1
        offset = (0.5, 0.0, 1.5) # (0, 0, 0) # (0.0, 0.0, -0.05)
        # shift = (np.sin(i * (2 * np.pi / interval)) * scale, np.cos(i * (2 * np.pi / interval)) * scale, 0)
        shift = (np.cos(i * (2 * np.pi / interval)) * scale, 0, 0)
        fov_scale = 1.0
        cam = Camera(uid=viewpoint_cam.uid,
                    fname=viewpoint_cam.fname,
                    R=viewpoint_cam.R,
                    T=np.array([viewpoint_cam.T[0]+offset[0]+shift[0], viewpoint_cam.T[1]+offset[1]+shift[1], viewpoint_cam.T[2]+offset[2]+shift[2]]),
                    FoVx=viewpoint_cam.FoVx*fov_scale,
                    FoVy=viewpoint_cam.FoVy*fov_scale,
                    cx=viewpoint_cam.cx,
                    cy=viewpoint_cam.cy,
                    image = None, mask = None, depth = None,
                    image_width=viewpoint_cam.image_width,
                    image_height=viewpoint_cam.image_height)
        dynamic_render_pkg = composite_render(cam, scene_gaussians, people_infos, pipe, background, d_color, d_opacity, 
                                                scaling_modifier = 1.0, override_color = None, render_only_people=False)
        dynamic_frame = dynamic_render_pkg["render"]
        dynamic_frame[dynamic_frame > 1] /= dynamic_frame[dynamic_frame > 1] # (turn off saturated points)
        save_rgb_image(dynamic_frame, path=dynamic_path / f"{i:05d}.png")
    video_name = f"dynamic_frame_{iteration:05d}.mp4"
    save_video(dynamic_path, video_name, fps=30)

@torch.no_grad()
def calculate_residuals_full(
    scene,
    people_infos,
    pipe,
    background,
    deform_pipe,
    full_cameras,
    delete_pid=None,
): 
    '''
    Calculate opacity and color residuals for all cameras.
    '''
    op_list = []
    col_list = []
    gmask_list = []
    # get opacity and color residuals for all cameras
    for cam in full_cameras:
        d_c, d_o = get_residuals(people_infos, deform_pipe, cam.fname, len(full_cameras))
        # mask out foreground area
        uv = composite_render(cam, scene.gaussians, people_infos, pipe, background, d_c, d_o, 
                                            scaling_modifier = 1.0, override_color = None, render_only_people=True, 
                                            delete_pid=delete_pid)["uv"]
        valid_uv = (
            (uv[:, 0] >= 0) & (uv[:, 0] < cam.image_height) &
            (uv[:, 1] >= 0) & (uv[:, 1] < cam.image_width)
        )
        if (not scene.fmask_dict is None) and (cam.fname in scene.fmask_dict):
            fore_mask = scene.fmask_dict[cam.fname].cuda()[None, None]
            fore_mask = torch.nn.functional.interpolate(fore_mask, (cam.image_height, cam.image_width), mode="bilinear", align_corners=False)
            fore_mask = fore_mask[0]
            fore_mask = 1 - fore_mask # reverse mask
            gs_mask = fore_mask[0, uv[:, 0].clamp(0, cam.image_height-1), \
                                        uv[:, 1].clamp(0, cam.image_width-1)] == 1 # (N,)
        else:
            gs_mask = torch.ones(uv.shape[0], dtype=torch.bool, device=uv.device)
        gs_mask = gs_mask * valid_uv # (N,)
        
        op_list.append(d_o)
        col_list.append(d_c)
        gmask_list.append(gs_mask)
    
    op_list = torch.stack(op_list, dim=0) # (B, N, 1)
    col_list = torch.stack(col_list, dim=0) # (B, N, 3)
    gmask_list = torch.stack(gmask_list, dim=0).unsqueeze(-1) # (B, N, 1)
    
    # if delete_pid is not None
    gs_offset = 0
    gs_mask = torch.ones(op_list.shape[1], dtype=torch.bool, device=op_list.device)
    for pnum, person_info in enumerate(people_infos):
        n_points = person_info.gaussians.get_n_points
        if pnum == delete_pid:
            gs_mask[gs_offset:gs_offset+n_points] = False
        gs_offset += n_points
    gs_mask = gs_mask.repeat(op_list.shape[0], 1).unsqueeze(-1) # (B, N, 1)
    op_list = op_list[gs_mask].view(op_list.shape[0], -1, 1) # (B, M, 1)
    col_list = col_list[gs_mask.expand(-1, -1, 3)].view(col_list.shape[0], -1, 3) # (B, M, 3)

    # mask out area occluded by foreground
    masked_opacity = op_list.masked_fill(~gmask_list, 0.0)  # (B, N, 1)
    masked_color   = col_list.masked_fill(~gmask_list, 0.0)  # (B, N, 3)
    count_per_n = gmask_list.sum(dim=0).clamp(min=1) # (N, 1)

    # calculate mean opacity and color
    mean_opacity = masked_opacity.sum(dim=0) / count_per_n # (N, 1)
    mean_color = masked_color.sum(dim=0) / count_per_n # (N, 3)

    _return = {
        "mean_opacity": mean_opacity,
        "mean_color": mean_color,
        "gmask_list": gmask_list,
        "op_list": op_list,
        "col_list": col_list,
    }
    return _return
