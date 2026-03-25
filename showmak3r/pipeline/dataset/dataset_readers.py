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
import sys
import glob
import json
import pickle5
from pathlib import Path
from typing import NamedTuple, Optional
from plyfile import PlyData, PlyElement


import cv2
import pandas
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from showmak3r.pipeline.scene.gaussian_model import BasicPointCloud, GaussianModel
from showmak3r.pipeline.dataset.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from showmak3r.utils.system_utils import searchForMaxIteration
from showmak3r.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from showmak3r.utils.sh_utils import SH2RGB


from config import stage


class CameraInfo(NamedTuple):
    uid: int
    fname: str
    R: np.array
    T: np.array
    FoVx: np.array
    FoVy: np.array
    image: np.array
    image_path: str
    mask: Optional[np.array] 
    mask_path: Optional[str] 
    fmask: Optional[np.array] 
    fmask_path: Optional[str] 
    depth: Optional[np.array]
    depth_path: Optional[str]
    width: int
    height: int
    cx: float
    cy: float
    smpl_param: Optional[np.array]
    bbox: Optional[list]

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info, radius_threshold=50):
    def get_center_and_diag(cam_centers):
        filtered_cc = [cam_center for cam_center in cam_centers if np.max(np.abs(cam_center))>=radius_threshold]
        print(f"filtered out {len(filtered_cc)} cameras")
        cam_centers = [cam_center for cam_center in cam_centers if np.max(np.abs(cam_center))<radius_threshold] # filtering
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    # single camera case
    if radius < 1e-3: 
        radius = 1
    print(f"scene radius: {radius}") # radius determine the "scene scale" -> important when densify / pruning

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, masks_folder=None, depths_folder=None):
    img_dict = dict()
    img_files = Path(images_folder).glob("*.png")
    for img_file in img_files:
        img_dict[img_file.name.split(".")[0]] = str(img_file) # fname
    
    cc_dict = dict()
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()
        
        # get camera info
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        
        cx = 0. # width / 2
        cy = 0. # height / 2

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FoVy = focal2fov(focal_length_x, height)
            FoVx = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FoVy = focal2fov(focal_length_y, height)
            FoVx = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        if not os.path.exists(image_path):
            fname = str(image_name)
            image_path = img_dict[fname]
        else:
            fname = str(image_name)
        image = Image.open(image_path)

        if masks_folder is not None:
            mask_path = Path(masks_folder) / f"{fname}.png.png" # colmap has .png.png format
            if mask_path.exists():
                mask = Image.open(mask_path)
            else:
                mask = None
                mask_path = None
        else:
            mask = None
            mask_path = None

        if depths_folder is not None:
            depth_path = Path(depths_folder) / f"{fname}_aligned.npy"
            if depth_path.exists():
                depth = np.load(depth_path)
            else:
                depth_path = None
                depth = None
        else:
            depth_path = None
            depth = None

        cam_info = CameraInfo(uid=uid, fname=fname, R=R, T=T, FoVy=FoVy, FoVx=FoVx, 
                            image_path=image_path, image=image, mask_path=mask_path, mask=mask, 
                            fmask_path=None, fmask=None, depth_path=depth_path, depth=depth, 
                            width=width, height=height, cx=cx, cy=cy, smpl_param=None, bbox=None)
        cam_infos.append(cam_info)

        # calculate cam_center
        c2w = np.eye(4)
        c2w[:3,:3] = R.T
        c2w[:3,3] = T
        w2c = np.linalg.inv(c2w)
        cc = w2c[:3, 3]

        cc_dict[key] = cc

    sys.stdout.write('\n')
    return cam_infos, cc_dict

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, normals=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    
    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    '''
    - from colmap_loader.py
    - read scene info from colmap data
    '''
    try:
        cameras_extrinsic_file = os.path.join(path, "undistorted/sparse", "0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "undistorted/sparse", "0","cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "undistorted/sparse", "0","images.txt")
        cameras_intrinsic_file = os.path.join(path, "undistorted/sparse", "0","cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted, cc_dict = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                                    images_folder=os.path.join(path, reading_dir), 
                                                    masks_folder=os.path.join(path, "colmap_masks"),
                                                    depths_folder=os.path.join(path, "depths", "inpainted_aligned")
                                                    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.fname)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0] # 80%
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0] # 20%
    else:
        train_cam_infos = cam_infos # 100%
        test_cam_infos = [] # 0%

    nerf_normalization = getNerfppNorm(train_cam_infos) # translate, radius

    ply_path = os.path.join(path, "undistorted/sparse/0/points3D.ply")
    bin_path = os.path.join(path, "undistorted/sparse/0/points3D.bin")
    txt_path = os.path.join(path, "undistorted/sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _, normal = read_points3D_text(txt_path, cc_dict=cc_dict, get_normal=True)
        except:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            normal = None
        storePly(ply_path, xyz, rgb, normals=normal)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCompositeSceneInfo(
        source_path, model_path, eval, 
        llffhold=8, 
        train_sample_interval: int=-1
        ):
    # load frames, masks, foreground masks and depths from data_path
    img_dir = os.path.join(source_path, 'composite', 'images')
    mask_dir = os.path.join(source_path, 'composite', 'actor_masks')
    fmask_dir = os.path.join(source_path, 'composite', 'foreground_masks')
    depth_dir = os.path.join(source_path, 'composite', 'depths', "original_aligned")

    img_dict = dict()
    mask_dict = dict()
    fmask_dict = dict()
    depth_dict = dict()
    assert os.path.exists(img_dir)
    for img_fname in (list(Path(img_dir).glob("*.png")) + list(Path(img_dir).glob("*.jpg"))):
        fname = str(os.path.basename(img_fname).split(".")[0])
        img_dict[fname] = str(img_fname)
    if not (mask_dir is None):
        for mask_fname in (list(Path(mask_dir).glob("*.png")) + list(Path(mask_dir).glob("*.jpg"))):
            fname = str(mask_fname.name.split(".")[0])
            mask_dict[fname] = str(mask_fname)
    if not (fmask_dir is None):
        for fmask_fname in (list(Path(fmask_dir).glob("*.png")) + list(Path(fmask_dir).glob("*.jpg"))):
            fname = str(fmask_fname.name.split(".")[0])
            fmask_dict[fname] = str(fmask_fname)
    if not (depth_dir is None):
        for depth_fname in (list(Path(depth_dir).glob("*.npy"))):
            fname = str(depth_fname.name.split("_aligned")[0])
            depth_dict[fname] = str(depth_fname)

    # load cameras from model_path
    if (Path(model_path) / 'actor' / 'cameras.pkl').exists(): 
        cam_file_path = str((Path(model_path) / 'actor' / 'cameras.pkl'))
    else:
        first_pnum = sorted(os.listdir(Path(model_path) / 'actor'))[0] # 001
        print(f"[INFO] Using actor {first_pnum} pickle file to load cameras.")
        cam_file_path = str((Path(model_path) / 'actor' / first_pnum / 'optimized.pkl'))

    # read camera datas
    try:
        cam_data = pandas.read_pickle(cam_file_path)
    except:
        import pickle5
        with open(cam_file_path, 'rb') as f:
            cam_data = pickle5.load(f)

    cam_infos_unsorted = []
    for uid, fname in enumerate(sorted(list(cam_data.keys()))):
        # sample frames per train_sample_interval
        if uid % train_sample_interval != 0:
            continue
        
        cam_dict = cam_data[fname]['camera']
        
        # load image
        image_path = img_dict[fname]
        image = Image.open(image_path)
        
        # get camera parameters
        R = np.transpose(cam_dict['rotation'])
        T = cam_dict['translation'].reshape(-1)
        focal_length_x = cam_dict['intrinsic'][0,0]
        focal_length_y = cam_dict['intrinsic'][1,1]
        if 'width' in cam_dict:
            width = cam_dict['width']
            height = cam_dict['height']
        else:
            width = image.size[0]
            height = image.size[1]
        FoVx = focal2fov(focal_length_x, width)
        FoVy = focal2fov(focal_length_y, height)
        cx = (cam_dict['intrinsic'][0,2] / (width / 2)) - 1.
        cy = (cam_dict['intrinsic'][1,2] / (height / 2)) - 1.
        
        if len(mask_dict) > 0 and fname in mask_dict:
            mask_path = mask_dict[fname] 
            mask = Image.open(mask_path)
        else:
            mask_path = None
            mask = None

        if len(fmask_dict) > 0 and fname in fmask_dict:
            fmask_path = fmask_dict[fname]
            fmask = Image.open(fmask_path)
        else:
            fmask_path = None
            fmask = None
        
        if len(depth_dict) > 0 and fname in depth_dict:
            depth_path = depth_dict[fname]
            depth = np.load(depth_path)
        else:
            depth_path = None
            depth = None

        cam_info = CameraInfo(uid=uid, fname=fname, R=R, T=T, FoVy=FoVy, FoVx=FoVx, 
                            image_path=image_path, image=image, mask_path=mask_path, mask=mask, 
                            fmask_path=fmask_path, fmask=fmask, depth_path=depth_path, depth=depth, 
                            width=width, height=height, cx=cx, cy=cy, smpl_param=None, bbox=None)
        cam_infos_unsorted.append(cam_info)
        
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.fname)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0] # 80%
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0] # 20%
    else:
        train_cam_infos = cam_infos # 100%
        test_cam_infos = [] # 0%
    
    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None)

    return scene_info

def readHumanSceneInfo(
    human_path, 
    eval, 
    scene_fnames=[], 
    person_mask_path: str="", 
    ):    
    # load canonical actor as point cloud
    ply_path = str(human_path.parent / f"canonical.ply")
    txt_path = str(human_path.parent / f"canonical.txt")
    xyz, rgb, _, normal = read_points3D_text(txt_path, get_normal=True)
    storePly(ply_path, xyz, rgb, normals=normal)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    
    # load masks if exists
    if person_mask_path != "":
        mask_list = glob.glob(os.path.join(person_mask_path, "*.png"))
        mask_list_jpg = glob.glob(os.path.join(person_mask_path, "*.jpg"))
        mask_list = sorted(list(mask_list)+list(mask_list_jpg))
        mask_dict = dict()
        for mask_fname in mask_list:
            fname = str(os.path.basename(mask_fname).split(".")[0])
            mask_dict[fname] = mask_fname
   
    # load cameras and smpls
    try:
        optimized_data = pandas.read_pickle(human_path)
    except:
        import pickle5
        with open(human_path, 'rb') as f:
            optimized_data = pickle5.load(f)

    default_image = np.zeros((512, 512, 3))
    cam_infos_unsorted = []
    for uid, fname in enumerate(sorted(list(optimized_data.keys()))):
        if len(scene_fnames) > 0:
            if fname not in scene_fnames:
                print(f"{fname} not in scene_fnames")
                continue
            
        # load optimized data of an actor
        v = optimized_data[fname]
        cam_dict = v['camera']
        smpl_param = v['smpl_param']

        # get camera parameters
        R = np.transpose(cam_dict['rotation'])
        T = cam_dict['translation'].reshape(-1)
        focal_length_x = cam_dict['intrinsic'][0,0]
        focal_length_y = cam_dict['intrinsic'][1,1]
        if 'width' in cam_dict:
            width = cam_dict['width']
            height = cam_dict['height']
        else:
            width = default_image.shape[1]
            height = default_image.shape[0]
        
        # get bbox
        if v['gt_bbox'] is None or 'gt_bbox' not in v:
            FoVy = focal2fov(focal_length_y, height)
            FoVx = focal2fov(focal_length_x, width)
            cx = (cam_dict['intrinsic'][0,2] / (width / 2)) - 1.
            cy = (cam_dict['intrinsic'][1,2] / (height / 2)) - 1.
            bbox = None
        else:
            bbox_dilation = 1.1

            gt_bbox = v['gt_bbox']
            b_x, b_y, b_w, b_h = gt_bbox
            b_x = int(b_x - b_w * (bbox_dilation-1)/2)
            b_y = int(b_y - b_h * (bbox_dilation-1)/2)
            b_w = int(b_w * bbox_dilation)
            b_h = int(b_h * bbox_dilation)
            

            b_x = 0 if b_x < 0 else b_x
            b_y = 0 if b_y < 0 else b_y
            b_w = width-b_x-1 if b_w + b_x >= width else b_w
            b_h = height-b_y-1 if b_h + b_y >= height else b_h

            new_cx = -(b_x) + v['camera']['intrinsic'][0,2]
            new_cy = -(b_y) + v['camera']['intrinsic'][1,2]

            # change intrinsic
            FoVy = focal2fov(focal_length_y, b_h)
            FoVx = focal2fov(focal_length_x, b_w)
            cx = (new_cx / (b_w / 2)) - 1.
            cy = (new_cy / (b_h / 2)) - 1.
            bbox = [b_x, b_y, b_w, b_h]

            # Finally change height and width
            width = b_w
            height = b_h
     
        # to save memory load dummy image in HumanScene cameras
        dummy_image = np.zeros((16, 16, 3))
        dummy_image = Image.fromarray(cv2.cvtColor(dummy_image.astype(np.uint8), cv2.COLOR_BGR2RGB))

        if person_mask_path != "":
            if fname not in mask_dict:
                mask_path = None
                mask = None
            else:
                mask_path = mask_dict[str(fname)] 
                mask = Image.open(mask_path)
        else:
            mask_path = None
            mask = None
        
        cam_info = CameraInfo(uid=uid, fname=fname, R=R, T=T, FoVy=FoVy, FoVx=FoVx, 
                            image_path=None, image=dummy_image, mask_path=mask_path, mask=mask, 
                            fmask_path=None, fmask=None, depth_path=None, depth=None, 
                            width=width, height=height, cx=cx, cy=cy, smpl_param=smpl_param, bbox=bbox)
        cam_infos_unsorted.append(cam_info)
        
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.fname)

    if eval:
        train_cam_infos = cam_infos
        test_cam_infos = []
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Composite": readCompositeSceneInfo,
    "Human": readHumanSceneInfo,
}
