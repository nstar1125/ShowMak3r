import numpy as np
import torch
from showmak3r.utils.loss_utils import gmof, l1_loss
from torch_scatter import scatter_min, scatter_max
import cv2
import os
import sys
import torch.nn as nn
RENDER_DEPTH_SAMPLE = False # for debugging

DEFAULT_DTYPE = torch.float32

def calculate_all_loss(
        cfg, num_all_frames, valid_frames,
        translation_params, rotation_params, scale_param, pose_params, beta_params,
        keypoint_jnts, keypoint_confs, bbox_scales, depth_maps, stage_depths, actor_masks,
        smpl_model, batch_cams, device
):
    num_valid_frames = len(valid_frames)
    scale_params = scale_param.reshape(1, 1).repeat(num_valid_frames, 1)
    
    smpl_params = torch.cat([
        scale_params,
        translation_params,
        rotation_params,
        pose_params,
        beta_params
    ], dim=-1) # 1+3+3+69+10 = 86

    smpl_outputs = smpl_model(smpl_params)
    world_smpl_jnts = smpl_outputs['smpl_jnts']
    world_smpl_verts = smpl_outputs['smpl_verts']

    # smpl to openpose format
    smpl2op_mapping = torch.tensor([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 32, 33, 34, 29, 30, 31], dtype=torch.long).to(device)
    world_smpl_j3d = torch.index_select(world_smpl_jnts, 1, smpl2op_mapping)
    
    # project to camera space
    cam_smpl_j3d = batch_cams(world_smpl_j3d)
    cam_smpl_verts = batch_cams(world_smpl_verts)
    
    loss_dict = dict()
    # fitting losses
    loss_dict['reprojection_loss'] = reproject_loss(cfg, keypoint_jnts, keypoint_confs, cam_smpl_j3d[:, :, :2], bbox_scales, device)
    loss_dict['depth_loss'] = depth_loss(cfg, depth_maps, cam_smpl_verts, device)
    loss_dict['trajectory_loss'] = trajectory_loss(cfg, num_all_frames, valid_frames, world_smpl_j3d, device)
    loss_dict['contact_loss'] = contact_loss(cfg, stage_depths, actor_masks, cam_smpl_verts, device)

    return loss_dict

def reproject_loss(cfg, keypoint_jnts, keypoint_confs, cam_smpl_j2d, bbox_scales, device): # reprojection loss with bbox scale as weight
    """
    Compares projected SMPL joints with detected 2D joints.
    """
    ignore_joints = [1,9,12] # neck, lr hip (due to estimator ambiguity)
    joint_weights = torch.ones(keypoint_jnts.shape[1])
    joint_weights[ignore_joints] = 0
    joint_weights = joint_weights.reshape((-1,1)).to(device)

    scaleFactor = 1. # 112 * 2      # -112 and 112 space originally used in SIMPLify, to balance with other terms. 
    denom = bbox_scales.reshape(-1, 1, 1).clamp_min(1e-6)
    joint_diff = scaleFactor * (keypoint_jnts - cam_smpl_j2d) / denom # weight joint_loss depending on bbox scale
    joint_diff = gmof(joint_diff, 100) # (B, J, 2)

    if (keypoint_confs.shape[-1] == joint_weights.shape[0]): # J matches
        _joint_weights = (keypoint_confs*joint_weights[:, 0]).unsqueeze(-1) ** 2
        joints_2dloss = (_joint_weights * joint_diff).sum(dim=[1,2]) / (_joint_weights.sum(dim=[1,2]) + 1e-9)
        joints_2dloss = joints_2dloss[_joint_weights.sum([1,2]) > 0] 
        joints_2dloss = joints_2dloss * joint_weights.sum()
    else:
        raise AssertionError() 
    
    joints_2dloss*=cfg.lambda_reproj
    return joints_2dloss.sum() # need to be accurate so (gmof(L2) + sum)

def depth_loss(cfg, depth_maps, cam_smpl_verts, device): # bbox_scales, 
    """
    Compares visible SMPL vertices with corresponding aligned depth map.
    """
    H, W, = depth_maps.shape[1:3]
    num_frames, num_points, _ = cam_smpl_verts.shape

    num_pixels = H * W
    smpl_verts_x = cam_smpl_verts[:, :, 0].type(torch.int64).clamp(0, W - 1)
    smpl_verts_y = cam_smpl_verts[:, :, 1].type(torch.int64).clamp(0, H - 1)
    smpl_verts_z = cam_smpl_verts[:, :, 2]

    background_mask = depth_maps==float('inf')
    smpl_map = torch.full((num_frames, num_pixels), float('inf'), device=device)
    indices = (smpl_verts_y * W + smpl_verts_x).view(num_frames, num_points)
    smpl_map, _ = scatter_min(smpl_verts_z.squeeze(-1), indices, out=smpl_map, dim=1)
    smpl_map = smpl_map.view(num_frames, H, W).unsqueeze(-1)
    smpl_map[background_mask] = float('inf')

    smpl_map = smpl_map.view(num_frames, num_pixels)
    gt_depths_map = depth_maps.view(num_frames, num_pixels)

    current_depths, current_indices = torch.topk(smpl_map, k=cfg.sample_num, dim=1, largest=False)  # sample closest
    pseudo_gt_depths = torch.gather(gt_depths_map, dim=1, index=current_indices)
    
    current_depths[pseudo_gt_depths == torch.inf] = 0.
    pseudo_gt_depths[pseudo_gt_depths == torch.inf] = 0.

    current_depths = current_depths.mean(dim=1)
    pseudo_gt_depths = pseudo_gt_depths.mean(dim=1)

    if RENDER_DEPTH_SAMPLE: # debug depth sampling
        for fidx in range(num_frames):
            depth_maps[background_mask] = 0.
            np_gt = depth_maps.view(num_frames, H, W).detach().cpu().numpy()
            
            normalized_gt = ((np_gt[fidx] - np.min(np_gt[fidx])) / (np.max(np_gt[fidx]) - np.min(np_gt[fidx])) * 255).astype(np.uint8)
            
            for k in range(cfg.sample_num):
                min_index_flat = current_indices[fidx, k].item()
                min_y, min_x = np.unravel_index(min_index_flat, (H, W))
                cv2.circle(normalized_gt, (min_x, min_y), radius=10, color=(255, 0, 0), thickness=2)
            
            cv2.imwrite("./debug_depth.png", normalized_gt)
        breakpoint()
    
    depth_loss = torch.abs(current_depths - pseudo_gt_depths)
    depth_loss*=cfg.lambda_depth
    return depth_loss.mean() # need to be robust so (L1 + mean)

def trajectory_loss(cfg, total_frames_num, valid_frames, world_smpl_j3d, device):
    """
    Compares SMPL joints with neighboring frames to smooth the trajectory using jerk loss.
    """
    fids = np.arange(total_frames_num)

    if len(fids) <= 3: # less than 3 frames
        return torch.zeros(1, dtype=torch.float32, requires_grad=True).sum().to(device)
    
    valid_fids = [int(fname.split('_')[1])-int(valid_frames[0].split('_')[1]) for fname in valid_frames]
    valid_mask = np.zeros(total_frames_num, dtype=int)
    valid_mask[valid_fids] = 1

    # calculate padded valid mask for occlusion handling
    padded_valid_mask = valid_mask.copy()
    zero_indices = np.where(valid_mask == 0)[0]
    
    # add 1 padding before zeros
    before_pad = zero_indices - 1
    before_pad = before_pad[before_pad >= 0]
    padded_valid_mask[before_pad] = 0
    
    # add 2 padding after zeros
    after_pad1 = zero_indices + 1
    after_pad1 = after_pad1[after_pad1 < total_frames_num]
    after_pad2 = zero_indices + 2
    after_pad2 = after_pad2[after_pad2 < total_frames_num]
    padded_valid_mask[after_pad1] = 0
    padded_valid_mask[after_pad2] = 0
    
    total_world_smpl_j3d = torch.zeros((total_frames_num, 25, 3), dtype=torch.float32, requires_grad=True).to(device)
    total_world_smpl_j3d[torch.tensor(valid_mask) == 1] = world_smpl_j3d
    
    # jerk loss
    total_world_smpl_j3d_t0 = torch.cat([total_world_smpl_j3d, 
                                    total_world_smpl_j3d[-1:].repeat(3, 1, 1)], dim=0)
    total_world_smpl_j3d_t1 = torch.cat([total_world_smpl_j3d[0:1],
                                    total_world_smpl_j3d, 
                                    total_world_smpl_j3d[-1:].repeat(2, 1, 1)], dim=0)
    total_world_smpl_j3d_t2 = torch.cat([total_world_smpl_j3d[0:1].repeat(2, 1, 1), 
                                    total_world_smpl_j3d, 
                                    total_world_smpl_j3d[-1:]], dim=0)                                                    
    total_world_smpl_j3d_t3 = torch.cat([total_world_smpl_j3d[0:1].repeat(3, 1, 1), 
                                    total_world_smpl_j3d], dim=0)
    acc1 = ((total_world_smpl_j3d_t0 + total_world_smpl_j3d_t1) \
                                - 2*total_world_smpl_j3d_t2)[1:-2]
    acc2 = ((total_world_smpl_j3d_t1 + total_world_smpl_j3d_t2) \
                                - 2*total_world_smpl_j3d_t3)[1:-2]

    traj_loss = (((acc1 - acc2)**2).mean(dim=(1,2)))
    traj_loss = traj_loss[torch.tensor(padded_valid_mask).to(device) == 1]
    traj_loss*=cfg.lambda_traj
    traj_loss = torch.nan_to_num(traj_loss, nan=0.0, posinf=0.0, neginf=0.0)
    return traj_loss.sum() # need to be accurate so (L2 + sum)

def contact_loss(cfg, stage_depths, actor_masks, cam_smpl_verts, device):
    """
    Regularizes fitting process if the actor penetrates the stage.
    """
    H, W, = stage_depths.shape[1:3]
    num_frames, num_points, _ = cam_smpl_verts.shape

    num_pixels = H * W
    smpl_verts_x = cam_smpl_verts[:, :, 0].type(torch.int64).clamp(0, W - 1)
    smpl_verts_y = cam_smpl_verts[:, :, 1].type(torch.int64).clamp(0, H - 1)
    smpl_verts_z = cam_smpl_verts[:, :, 2]
    
    # generate smpls map
    smpl_map = torch.full((num_frames, num_pixels), float('-1.'), device=device)
    indices = (smpl_verts_y * W + smpl_verts_x).view(num_frames, num_points)
    smpl_map, _ = scatter_max(smpl_verts_z.squeeze(-1), indices, out=smpl_map, dim=1)
    smpl_map = smpl_map.view(num_frames, H, W)
    vertices_bg_mask = smpl_map == float('-1.')
    vertices_bg_mask = vertices_bg_mask * actor_masks # only visible points
    vertices_bg_mask = vertices_bg_mask.bool()
    
    smpl_map[vertices_bg_mask] = 0.
    _stage_depths = stage_depths.clone()
    _stage_depths[vertices_bg_mask] = 0.
    
    denom = (~vertices_bg_mask).sum(dim=(1,2)).clamp_min(1)
    contact_loss = torch.clamp(smpl_map - _stage_depths, min=0).sum(dim=(1, 2))/denom
    contact_loss*=cfg.lambda_contact
    
    return contact_loss.sum()