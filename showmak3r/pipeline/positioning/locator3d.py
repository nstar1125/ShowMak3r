import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from showmak3r.pipeline.positioning.locator3d_loss import calculate_all_loss
# from showmak3r.utils.plot_utils import plot_2D_trajectory, plot_3D_trajectory

from pathlib import Path
from scipy.spatial.transform import Rotation as Rot, Slerp

class BatchProjectionCameras(nn.Module):
    '''
    Project 3D points to 2D image points in batch.
    '''
    def __init__(self, rotations, translations,
                 batch_fx, batch_fy, centers, device,
                 dtype=torch.float32):
        super(BatchProjectionCameras, self).__init__()
        B = rotations.shape[0]
        self.cam_proj_matrix = torch.eye(4, device=device, dtype=dtype).repeat(B, 1, 1)
        self.cam_proj_matrix[:, :3, :3] = rotations
        self.cam_proj_matrix[:, :3, 3] = translations
        
        self.cam_focal_matrix = torch.eye(2, device=device, dtype=dtype).repeat(B, 1, 1)
        self.cam_focal_matrix[:, 0, 0] = batch_fx
        self.cam_focal_matrix[:, 1, 1] = batch_fy
        self.centers = centers

        c2w_rotations = torch.transpose(rotations, 1, 2)
        self.global_cam_positions = torch.einsum('bij,bj->bi', c2w_rotations, translations)

        self.device = device

    def forward(self, batch_points):
        # project points to camera space
        batch_points_homo = torch.cat([batch_points, torch.ones_like(batch_points[..., :1])], dim=-1)
        batch_proj_points = torch.einsum('bki,bji->bjk',[self.cam_proj_matrix, batch_points_homo])
        
        # get image points
        batch_img_points = torch.div(batch_proj_points[:, :, :2], batch_proj_points[:, :, 2].unsqueeze(dim=-1))
        batch_img_points = torch.einsum('bki,bji->bjk', [self.cam_focal_matrix, batch_img_points]) \
            + self.centers.unsqueeze(dim=1)
        
        return torch.cat([batch_img_points, batch_proj_points[:, :, 2, None]], dim=-1)

# ================================================================================

def interpolate_frames(cfg, person_dict, start_fid, num_frames, mode):
    '''
    pre: True = only smoothing and no interpolation
    post: False = only interpolation and no smoothing
    '''
    def bidirectional_filter(data, alpha=0.1):
        # Forward filtering
        filtered_data_forward = np.zeros_like(data)
        filtered_data_forward[0] = data[0]
        
        for t in range(1, len(data)):
            filtered_data_forward[t] = alpha * data[t] + (1 - alpha) * filtered_data_forward[t - 1]
        
        # Backward filtering
        filtered_data_backward = np.zeros_like(data)
        filtered_data_backward[-1] = filtered_data_forward[-1]
        
        for t in range(len(data) - 2, -1, -1):
            filtered_data_backward[t] = alpha * filtered_data_forward[t] + (1 - alpha) * filtered_data_backward[t + 1]

        return filtered_data_backward

    def interpolate_missing(params, is_bodypose=False):
        param_values = params
        valid_idx = ~np.isnan(param_values)
        nans_idx = np.isnan(param_values)
        
        if np.all(nans_idx):
            param_values[nans_idx] = 0.
            return param_values # can not interpolate if all values are missing
        
        if np.any(nans_idx):  # If there are missing values, interpolate first
            param_values[nans_idx] = np.interp(
                fids[nans_idx],
                fids[valid_idx],
                param_values[valid_idx]
            )    
        
        if mode == 'pre': # smoothing to enhance frame consistency / no smoothing after fitting (can degrade actor training)
            param_values = bidirectional_filter(param_values, alpha=cfg.smooth_alpha) # lower alpha = more smoothing
        
        if is_bodypose: # do not interpolate body pose
            param_values[nans_idx] = 0.
        
        return param_values

    # ------------------------------ fill in missing frames ------------------------------

    assert mode in ['pre', 'post']
    valid_frames = sorted(person_dict.keys()) # invalid frames have been removed
    bbox_params = []
    smpl_params = []
    # j3ds_params = [] 
    j2ds_params = [] 
    dwp_jnts_params = [] 
    left_hand_params = []
    right_hand_params = []
    face_params = []
    for fid in range(start_fid, start_fid+num_frames):
        fname = f"frame_{fid+1:04d}"
        if fname in valid_frames: # skip hands and faces
            bbox_params.append(person_dict[fname]['bbox']) # (4,)
            smpl_params.append(person_dict[fname]['smpl_param']) # (86,)
            j2ds_params.append(person_dict[fname]['j2d']) # (25, 2)
            dwp_jnts_params.append(person_dict[fname]['body']) # (25, 3)
        else:
            bbox_params.append(torch.full((4,), np.nan, dtype=torch.float32))
            smpl_params.append(torch.full((86,), np.nan, dtype=torch.float32))
            j2ds_params.append(torch.full((25, 2), np.nan, dtype=torch.float32))
            dwp_jnts_params.append(torch.full((25, 3), np.nan, dtype=torch.float32))
    
    bbox_tensor = torch.stack(bbox_params, dim=0)
    smpl_tensor = torch.stack(smpl_params, dim=0)
    j2ds_tensor = torch.stack(j2ds_params, dim=0)
    dwp_jnts_tensor = torch.stack(dwp_jnts_params, dim=0)

    # ------------------------------ interpolate and smoothing ------------------------------
    fids = np.arange(start_fid, start_fid+num_frames)
    
    # bbox_parameters
    for bbox_idx in range(4):
        bbox_tensor[:, bbox_idx] = torch.tensor(interpolate_missing(bbox_tensor[:, bbox_idx].numpy()))
    
    # SMPL_parameters
    for param_idx in range(86):  # Interpolate for each of the 86 parameters separately
        param_values = smpl_tensor[:, param_idx].numpy()
        if 4<=param_idx and param_idx<7: # skip rotation
            valid_idx = ~np.isnan(param_values)
            nans_idx = np.isnan(param_values)
            if param_idx == 4:
                axis_angle_params = smpl_tensor[:, 4:7].numpy()
                if np.any(nans_idx):
                    if np.all(nans_idx):
                        smpl_tensor[:, 4:7] = np.full(axis_angle_params.shape, 0.)
                        continue
                    # Rot to Quat
                    rotations = Rot.from_rotvec(axis_angle_params[valid_idx])
                    quaternions = rotations.as_quat()  # (N, 4)

                    # Slerp Interpolation
                    fids_valid = fids[valid_idx]
                    slerp = Slerp(fids_valid, Rot.from_quat(quaternions))

                    # Valid
                    in_range_idx = (fids >= fids_valid[0]) & (fids <= fids_valid[-1]) & nans_idx
                    if np.any(in_range_idx):
                        interpolated_rotations = slerp(fids[in_range_idx])
                        axis_angle_params[in_range_idx] = interpolated_rotations.as_rotvec()

                    # Front Extrapolation
                    front_nan_idx = (fids < fids_valid[0])
                    if np.any(front_nan_idx): 
                        front_rotation = rotations[0].as_rotvec()
                        axis_angle_params[front_nan_idx] = front_rotation

                    # Back Extrapolation
                    back_nan_idx = (fids > fids_valid[-1])
                    if np.any(back_nan_idx):
                        back_rotation = rotations[-1].as_rotvec()
                        axis_angle_params[back_nan_idx] = back_rotation
                    
                    smpl_tensor[:, 4:7] = torch.tensor(axis_angle_params)
            continue            
        smpl_tensor[:, param_idx] = torch.tensor(interpolate_missing(param_values))
    
    # comotion_j2ds_params: for parts where dwpose is not available
    for joint_idx in range(25):
        for coord_idx in range(2):
            j2ds_tensor[:, joint_idx, coord_idx] = torch.tensor(interpolate_missing(j2ds_tensor[:, joint_idx, coord_idx].numpy()))
    
    # body_pose
    for joint_idx in range(25):
        for coord_idx in range(3):
            dwp_jnts_tensor[:, joint_idx, coord_idx] = \
                torch.tensor(interpolate_missing(dwp_jnts_tensor[:, joint_idx, coord_idx].numpy(), is_bodypose=True))
    
    # ------------------------------ save result ------------------------------
    
    for i in range(num_frames):
        fid = start_fid + i
        fname = f"frame_{fid+1:04d}"
        if bbox_tensor[i, :].isnan().all() and \
            smpl_tensor[i, :].isnan().all() and \
            j2ds_tensor[i, :, :].isnan().all() and \
            dwp_jnts_tensor[i, :, :].isnan().all():
            continue # Skip if all frames in the shot was not detected
        
        if (mode == 'pre' and fname in person_dict.keys()) or (mode == 'post' and fname not in person_dict.keys()):
            person_dict[fname] = { 
                'bbox': bbox_tensor[i, :],
                'smpl_param': smpl_tensor[i, :],
                'j2d': j2ds_tensor[i, :, :],
                'body': dwp_jnts_tensor[i, :, :],
            }
    return person_dict

# ================================================================================

def log_results(cfg, pnum, iteration, total_loss, loss_dict, scale_param, data_params, stage_num):

    if cfg.use_wandb:
        if stage_num == 1:
            wandb.log({
                f"[{pnum}.{stage_num}] scale": scale_param.item(),
                f"[{pnum}.{stage_num}] total_loss": total_loss,
                f"[{pnum}.{stage_num}] reproj_loss": loss_dict['reprojection_loss'].sum(),
                f"[{pnum}.{stage_num}] depth_loss": loss_dict['depth_loss'].sum(),
                f"[{pnum}.{stage_num}] traj_loss": loss_dict['trajectory_loss'].sum(),
                f"[{pnum}.{stage_num}] contact_loss": loss_dict['contact_loss'].sum(),
            })
        elif stage_num == 2:
            wandb.log({
                f"[{pnum}.{stage_num}] total_loss": total_loss,
                f"[{pnum}.{stage_num}] reproj_loss": loss_dict['reprojection_loss'].sum(),
                f"[{pnum}.{stage_num}] depth_loss": loss_dict['depth_loss'].sum(),
                f"[{pnum}.{stage_num}] traj_loss": loss_dict['trajectory_loss'].sum(),
                f"[{pnum}.{stage_num}] contact_loss": loss_dict['contact_loss'].sum(),
            })
        


# ================================================================================

def fit_single_actor(
    cfg, pnum, start_fid, num_frames, 
    person_dict, cam_dicts, img_dict, mask_dict, depth_dict, stage_depth_dict, smpl_model, device
):
    """
    Fit single actor in a shot using depth guidance.

    Args:
        cfg: config
        pnum: actor number
        start_fid: start frame id of the shot
        num_frames: number of frames in the shot
        person_dict: person dictionary
        cam_dicts: camera dictionary of the shot
        img_dict: image dictionary of the shot
        mask_dict: mask dictionary of the shot
        depth_dict: depth dictionary of the shot
        stage_depth_dict: stage depth dictionary of the shot
    
    Returns:
        person_dict: person dictionary with fitted SMPL parameters
    """
    # detect missing frames
    detected_list = []
    for fid in range(start_fid, start_fid+num_frames):
        fname = f"frame_{fid+1:04d}"
        if fname not in depth_dict: # if no depth map, -> not detected
            detected_list.append(False)
        else:
            detected_list.append(fname in person_dict.keys())
    detected_frames = np.array(detected_list)
    missing_indices = np.where(detected_frames==False)[0]

    # expand missing frames for robust interpolation
    for missing_fid in missing_indices:
        start = max(start_fid, missing_fid - cfg.delete_range)
        end = min(start_fid + num_frames, missing_fid + cfg.delete_range + 1)
        detected_frames[start:end] = False
    
    # remove frames not detected
    for i, fid in enumerate(range(start_fid, start_fid+num_frames)):
        fname = f"frame_{fid+1:04d}"
        if fname in person_dict.keys() and not detected_frames[i]:
            person_dict.pop(fname)
            print(f"delete {fname} for robust interpolation")
    valid_frames = sorted(list(person_dict.keys()))

    # if no valid frames, return empty dict
    if len(valid_frames) == 0:
        return person_dict

    # ------------------------------ Prepare Cameras ------------------------------
    Rs = []
    Ts = []
    centers = []
    fxs = []
    fys = []
    radius_threshold = 100
    for fname in sorted(person_dict.keys()):
        assert fname in cam_dicts.keys()
        cam_dict = cam_dicts[fname]
        R = torch.from_numpy(cam_dict['w2c'][:3,:3]).float()
        T = torch.from_numpy(cam_dict['w2c'][:3,-1]).float()
        center = torch.tensor([cam_dict['cx'], cam_dict['cy']]).float()
        fx = torch.tensor(cam_dict['fx'], dtype=torch.float32)
        fy = torch.tensor(cam_dict['fy'], dtype=torch.float32)
        
        # remove out of bound camera
        R_c2w = R.T
        T_c2w = -R_c2w @ T
        if ((T_c2w**2).sum()**0.5) > radius_threshold:
            valid_frames.remove(fname)
            person_dict.pop(fname)
            print(f"delete {fname} due to out-of-bound camera")
            continue
        
        Rs.append(R)
        Ts.append(T)
        centers.append(center)
        fxs.append(fx)
        fys.append(fy)
    
    rotation_tensor = torch.stack(Rs).to(device)
    translation_tensor = torch.stack(Ts).to(device)
    fx_tensor = torch.stack(fxs).to(device)
    fy_tensor = torch.stack(fys).to(device)
    center_tensor = torch.stack(centers).to(device)
    
    # setup projection cameras
    batch_cams = BatchProjectionCameras(
        rotation_tensor,
        translation_tensor,
        fx_tensor,
        fy_tensor,
        center_tensor,
        device
    ) 

    # smooth parameters / X interpolation
    person_dict = interpolate_frames(cfg, person_dict, start_fid, num_frames, mode='pre')
    
    # ------------------------------ Prepare SMPL parameters ------------------------------
    # SMPL parameters
    smpl_scales = []
    smpl_trans = []
    smpl_poses = []
    smpl_shape = []
    
    # DWPose parameters
    dwp_jnts_stack = []
    dwp_confs_stack = []
    
    # Others
    bbox_scale_stack = []
    depth_map_stack = []
    stage_depth_stack = []
    mask_stack = []
    
    for fname in valid_frames:
        frame_dict = person_dict[fname]

        # collect DWPose GTs
        if frame_dict['body'] is None:
            dwp_jnts = torch.zeros((25, 2))
            dwp_confs = torch.zeros((25))
        else:
            dwp_jnts = []
            dwp_confs = []
            body = frame_dict['body']
            num_valid_jnts = 0
            for jnt in body:
                if jnt.isnan().all():
                    dwp_jnts.append(torch.zeros(2, dtype=torch.float32))
                    dwp_confs.append(torch.tensor(0, dtype=torch.float32))
                else:
                    dwp_jnts.append(jnt[:2])
                    if jnt[2] < cfg.joint_threshold:
                        conf = torch.tensor(0, dtype=torch.float32)
                    else:
                        conf = jnt[2]
                    dwp_confs.append(conf)
                    if conf >= cfg.joint_threshold:
                        num_valid_jnts += 1
            dwp_jnts = torch.stack(dwp_jnts, dim=0)
            dwp_confs = torch.stack(dwp_confs, dim=0)
            
            if num_valid_jnts < cfg.min_jnts_num: # if valid joints are below minimum joints number, exclude.
                print(f"skipping {fname} due to number of inaccurate joints (#: {num_valid_jnts})")
                dwp_confs *= 0
        
        smpl_scales.append(frame_dict['smpl_param'][0])
        smpl_trans.append(frame_dict['smpl_param'][1:4])
        smpl_poses.append(frame_dict['smpl_param'][4:-10])
        smpl_shape.append(frame_dict['smpl_param'][-10:])
        
        dwp_jnts_stack.append(dwp_jnts)
        dwp_confs_stack.append(dwp_confs)
        bbox_size = (frame_dict['bbox'][2] + frame_dict['bbox'][3]) / 2.
        bbox_scale_stack.append(bbox_size)
        depth_map_stack.append(torch.tensor(depth_dict[fname], dtype=torch.float32))
        stage_depth_stack.append(torch.tensor(stage_depth_dict[fname], dtype=torch.float32))
        mask_stack.append(torch.tensor(mask_dict[fname], dtype=torch.float32))

    smpl_scales_tensor = torch.stack(smpl_scales, dim=0).to(device)
    smpl_trans_tensor = torch.stack(smpl_trans, dim=0).to(device)
    smpl_poses_tensor = torch.stack(smpl_poses, dim=0).to(device)
    smpl_shape_tensor = torch.stack(smpl_shape, dim=0).to(device)
    
    dwp_jnts_stack = torch.stack(dwp_jnts_stack, dim=0).to(device)
    dwp_confs_stack = torch.stack(dwp_confs_stack, dim=0).to(device)
    
    bbox_scale_stack = torch.stack(bbox_scale_stack, dim=0).squeeze().to(device)
    depth_map_stack = torch.stack(depth_map_stack, dim=0).to(device)
    stage_depth_stack = torch.stack(stage_depth_stack, dim=0).to(device)
    mask_stack = torch.stack(mask_stack, dim=0).to(device)

    # set optimization targets
    scale_param = torch.tensor(smpl_scales_tensor.mean(dim=0).cpu().numpy(), dtype=torch.float32, requires_grad=True, device=device)
    translation_params = torch.tensor(smpl_trans_tensor.cpu().numpy(), dtype=torch.float32, requires_grad=True, device=device)
    rotation_params = torch.tensor(smpl_poses_tensor[:, :3].cpu().numpy(), dtype=torch.float32, requires_grad=True, device=device)
    pose_params = torch.tensor(smpl_poses_tensor[:, 3:].cpu().numpy(), dtype=torch.float32, requires_grad=True, device=device)
    beta_params = torch.tensor(smpl_shape_tensor.cpu().numpy(), dtype=torch.float32, requires_grad=True, device=device)

    # if all frames have invalid joints, return empty dict
    if dwp_jnts_stack.isnan().all() and dwp_confs_stack.isnan().all():
        return dict()
    # ------------------------------ 1. Fitting Process ------------------------------
    '''
    Stage 1. fit SMPLs to world space.
    
    Loss:
    - re-projection loss
    - depth loss
    - trajectory loss
    - contact loss
    
    Parameters:
    - translation, rotation, scale
    '''
    progress_bar = tqdm(range(cfg.stage1_iterations), desc=f"[Stage 1/2] Positioning person {pnum}")
    weight_dict = {
        "reprojection_loss" : lambda cst, it: cst,
        "depth_loss":  lambda cst, it: cst,
        "trajectory_loss":  lambda cst, it: cst,
        "contact_loss":  lambda cst, it: cst,
    } 
    target_params = [
        {'params': scale_param, 'lr': 0.01}, # 0.1
        {'params': translation_params, 'lr': 0.015}, # 0.2
        {'params': rotation_params, 'lr': 0.005},
    ]
    optimizer = torch.optim.Adam(target_params, lr=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999) # 0.998
    for iteration in progress_bar:
        if iteration > cfg.stage1_iterations - 1000:
            weight_dict["depth_loss"] = lambda cst, it: cst * 0.01

        optimizer.zero_grad()
        
        # calculate all losses
        loss_dict = calculate_all_loss(
            cfg, 
            num_frames, 
            valid_frames,
            translation_params, rotation_params, scale_param, pose_params, beta_params, 
            dwp_jnts_stack, dwp_confs_stack, bbox_scale_stack, depth_map_stack, stage_depth_stack, mask_stack,
            smpl_model, batch_cams, device
        )
        total_loss = 0.
        for k in weight_dict:
            if k not in loss_dict:
                continue
            total_loss += weight_dict[k](loss_dict[k], iteration)
        if total_loss.isnan().sum() > 0:
            raise AssertionError()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({"Loss": f"{total_loss:.4f}", "lr": scheduler.get_last_lr()[0]})
        
        # log
        log_results(cfg, pnum, iteration, total_loss, loss_dict, 
                    scale_param, translation_params, 1)

    # ------------------------------ 2. Fitting Process ------------------------------
    '''
    Stage 2. align SMPL keypoints to DWPose keypoints.
    
    Loss:
    - re-projection loss
    - angle prior loss
    - pose prior loss
    
    Parameters:
    - rotation, pose
    '''
    progress_bar = tqdm(range(cfg.stage2_iterations), desc=f"[Stage 2/2] Fitting person {pnum}")
    weight_dict = {
        "reprojection_loss" : lambda cst, it: cst,
        "trajectory_loss":  lambda cst, it: cst,
        "contact_loss":  lambda cst, it: cst,
    } 
    target_params = [
        {'params': rotation_params, 'lr': 5e-3},
        {'params': translation_params, 'lr': 1e-2},
    ]
    optimizer = torch.optim.Adam(target_params, lr=1e-2, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    for iteration in progress_bar:
        optimizer.zero_grad()
        
        # calculate all losses
        loss_dict = calculate_all_loss(
            cfg, 
            num_frames, 
            valid_frames,
            translation_params, rotation_params, scale_param, pose_params, beta_params, 
            dwp_jnts_stack, dwp_confs_stack, bbox_scale_stack, depth_map_stack, stage_depth_stack, mask_stack,
            smpl_model, batch_cams, device
        )
        total_loss = 0.
        for k in weight_dict:
            if k not in loss_dict:
                continue
            total_loss += weight_dict[k](loss_dict[k], iteration)
        if total_loss.isnan().sum() > 0:
            raise AssertionError()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({"Loss": f"{total_loss:.4f}", "lr": scheduler.get_last_lr()[0]})
        
        # log
        log_results(cfg, pnum, iteration, total_loss, loss_dict, 
                    scale_param, translation_params, 2)
    
    # ------------------------------ Save result ------------------------------

    valid_frames_num = len(valid_frames)
    fitted_smpl_tensor = torch.cat([
        scale_param.reshape(1, 1).repeat(valid_frames_num, 1),
        translation_params,
        rotation_params,
        pose_params,
        beta_params
    ], dim=-1)
    fitted_smpl_params = fitted_smpl_tensor.detach().cpu()

    for i, fname in enumerate(valid_frames):
        person_dict[fname]['smpl_param'] = fitted_smpl_params[i]

    # ------------------------------ Interpolate missing frames ------------------------------
    # Interpolation O / Smoothing X
    person_dict = interpolate_frames(cfg, person_dict, start_fid, num_frames, mode='post')

    return person_dict