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


from typing import Dict, Any, Optional

import json
import torch
import numpy as np
import math
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import List
from showmak3r.utils.general_utils import PILtoTorch, quaternion_rotation_y, quaternion_rotation_x, quaternion_rotation_z
from showmak3r.utils.graphics_utils import fov2focal, focal2fov, rotation_matrix_from_vectors
from showmak3r.pipeline.dataset.colmap_loader import qvec2rotmat
from showmak3r.pipeline.scene.cameras import Camera

WARNED = False
def loadCam(args, id, cam_info, resolution_scale, _resolution, data_device): 
    if _resolution is None:
        _resolution = args.resolution
    
    if data_device is None:
        data_device = args.data_device
    
    orig_w, orig_h = cam_info.width, cam_info.height

    if _resolution in [1, 2, 4, 8]:
        scale = (resolution_scale * _resolution)
        resolution = round(orig_w/scale), round(orig_h/scale)
    else:  # should be a type that converts to float
        if _resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / _resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    bbox = cam_info.bbox

    # load image
    if cam_info.image is None:
        resized_image_rgb = None
        gt_image = None
    else:
        if not (bbox is None):   
            bb = bbox
            _img = cam_info.image.crop((bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))
        else:
            _img = cam_info.image
        resized_image_rgb = PILtoTorch(_img, resolution) # C, H, W: 0~1 value range
        gt_image = resized_image_rgb[:3, ...]
    
    if cam_info.mask is None: # resize mask
        resized_mask = None
        gt_mask = None
    else:
        mask = cam_info.mask
        if args.dilate_ratio > 0:
            from showmak3r.utils.general_utils import erode_mask
            mask = np.asarray(mask)
            dilate_kernel_size = int((resolution[0] + resolution[1]) * args.dilate_ratio)
            mask = erode_mask(mask, kernel_size = dilate_kernel_size)
            mask = Image.fromarray(mask)

        if not bbox is None:
            mask = mask.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

        resized_mask = PILtoTorch(mask, resolution)
        gt_mask = resized_mask[:1, ...] # (1, resized_H, resized_W)

    # depth
    if cam_info.depth is None: # resize depth
        resized_depth = None
        gt_depth = None
    else:
        depth_tensor = torch.from_numpy(cam_info.depth).permute(2,0,1).float()
        resized_depth = torch.nn.functional.interpolate(
            depth_tensor.unsqueeze(1),
            size=(resolution[1], resolution[0]),
            mode="bicubic",
            align_corners=False,
        ).squeeze() 
        gt_depth = resized_depth.unsqueeze(0) # (1, resized_H, resized_W)

    # bbox
    if not (bbox is None):   
        bbox = [int(bb / scale) for bb in bbox]
    else:
        bbox = None
    
    # smpl
    if not cam_info.smpl_param is None:
        smpl_param = cam_info.smpl_param
    else:
        smpl_param = None
    
    return Camera(uid=cam_info.uid, fname=cam_info.fname, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FoVx, FoVy=cam_info.FoVy, cx=cam_info.cx, cy=cam_info.cy,
                  image=gt_image, mask=gt_mask, depth=gt_depth,
                  image_width=resolution[0], image_height=resolution[1], 
                  bbox=bbox, smpl_param=smpl_param, data_device=data_device)

def cameraList_from_camInfos(
        cam_infos, 
        resolution_scale, 
        args=None, 
        resolution=None, 
        data_device=None, 
        ):
    '''
    fetch camera list from cam_infos
    '''
    camera_list = []

    for id, cam_info in tqdm(enumerate(cam_infos), desc="Loading Cameras", total=len(cam_infos)):
        camera_list.append(loadCam(
                                    args, 
                                    id, 
                                    cam_info, 
                                    resolution_scale, 
                                    resolution, 
                                    data_device, 
                                    )
                           )

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : camera.uid,
        'img_name' : camera.fname,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FoVy, camera.height),
        'fx' : fov2focal(camera.FoVx, camera.width)
    }
    return camera_entry

def gen_canon_cams(n_cameras: int=36, t=3, f=700, res=512, device=torch.device("cuda"), rot_x=False, rot_z=False):
    fov = focal2fov(f, res)

    # Generate quaternion rotations
    angles_deg = np.arange(0, 361, 360//n_cameras)
    if rot_x:
        quaternions = [quaternion_rotation_x(angle) for angle in angles_deg]
    elif rot_z:
        quaternions = [quaternion_rotation_z(angle) for angle in angles_deg]
    else:
        quaternions = [quaternion_rotation_y(angle) for angle in angles_deg]

    org_image = torch.zeros(3, res, res)

    # Print the generated quaternions
    uid = 0
    cameras = []
    for angle, quat in zip(angles_deg, quaternions):
        # print(f"Angle: {angle} degrees, Quaternion: {quat}")
        R = qvec2rotmat(quat)

        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = np.array([0, 0.3, t])      # I shift 0.3 up in depth direction
        c2w = np.linalg.inv(w2c)

        c2w[0:3, 0] *= -1
        c2w[0:3, 1] *= -1    
        
        ext = np.linalg.inv(c2w)
        T = ext[:3, 3]
        R = ext[:3, :3].T

        cam = Camera(uid=0, fname=None, R=R, T=T, FoVx=fov, FoVy=fov, cx=0, cy=0, 
                     image=org_image, mask=None, depth=None,
                     smpl_param=None, data_device=device)

        uid += 1
        cameras.append(cam)

    return cameras



def gen_perturbed_camera(view, n_cameras: int=36, radius: float=0.01):
    angles_deg = np.arange(0, 361, 360//n_cameras)

    # get c2w
    c2w_R = view.R
    w2c_T = view.T

    # get cam_center
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = c2w_R.transpose()
    Rt[:3, 3] = w2c_T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    uid = view.uid
    colmap_id = view.colmap_id
    R = C2W[:3,:3]
    fovx = view.FoVx
    fovy = view.FoVy
    cx = view.cx
    cy = view.cy
    device = torch.device
    org_image = view.original_image
    data_device = view.data_device


    perturb_cams = []
    for angle in angles_deg:
        x = np.cos(angle*np.pi/180) * radius
        y = np.sin(angle*np.pi/180) * radius

        # New camera center in camer space
        new_center = np.array([x, y, 0])
        new_c2w = C2W.copy()
        new_c2w[:3, 3] += C2W[:3,:3] @ new_center

        T = np.linalg.inv(new_c2w)[:3, 3]
        
        
        cam = Camera(colmap_id=colmap_id, R=R, T=T, 
                  FoVx=fovx, FoVy=fovy, 
                  cx=cx, cy=cy,
                  image=org_image, gt_alpha_mask=None, depth=None,
                  image_name=None, uid=uid, data_device=data_device, smpl_param=None)

        perturb_cams.append(cam)
        
    return perturb_cams