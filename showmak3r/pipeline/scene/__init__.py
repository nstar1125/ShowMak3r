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
import random
import json
import pandas
import torch
import numpy as np

from typing import List, Optional
from pathlib import Path

from showmak3r.pipeline.dataset.dataset_readers import sceneLoadTypeCallbacks
from showmak3r.pipeline.scene.gaussian_model import GaussianModel
from showmak3r.utils.system_utils import searchForMaxIteration
from showmak3r.pipeline.dataset.camera_loader import cameraList_from_camInfos, camera_to_JSON
from config.stage import ModelParams

class Scene:
    '''
    load scene cameras and stage gaussians
    '''
    gaussians : GaussianModel
    def __init__(self, args : ModelParams, gaussians : GaussianModel, scene_type: str, load_iteration=None, 
                shuffle=False, resolution_scales=[1.0], fast_loader=False, view_dir_reg=False, init_opacity=0.1, 
                exp_name=None):

        self.init_w_normal = view_dir_reg
        self.model_path = args.model_path
        if not (exp_name is None):
            self.model_path = os.path.join(self.model_path, exp_name)
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "stage", "point_cloud")) 
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        if scene_type == "stage":
            assert os.path.exists(os.path.join(args.background_path, "undistorted/sparse"))
            print("Loading scene from COLMAP")
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.background_path, "images", args.eval)
        elif scene_type == "actor":
            print("Loading scene from Composite data")
            scene_info = sceneLoadTypeCallbacks["Composite"](args.source_path, args.model_path, args.eval)
        else:
            raise Exception('[WARNING] Dataloader not implemented!')
    
        if not self.loaded_iter:
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
                
            with open(os.path.join(args.background_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            if fast_loader:
                self.train_cameras = scene_info.train_cameras
            else:
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, 
                                                                                resolution_scale, 
                                                                                args=args, 
                                                                                data_device="cuda"
                                                                                ) 
            
            if len(scene_info.test_cameras) > 0:
                print("Loading Test Cameras")
            else:
                print("No Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, 
                                                                                resolution_scale, 
                                                                                args=args, 
                                                                                data_device="cuda")

        
        if self.loaded_iter:
            print("Loading Optimized Point Clouds")
            self.gaussians.load_ply(os.path.join(self.model_path, "stage", "point_cloud", f"iteration_{self.loaded_iter}.ply"))
        else:
            print("Loading Initial Point Clouds")
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.init_w_normal, init_opacity)
        
    # TODO: deprecated
    def save(self, iteration, smpl_params=None, deformer=None, people_infos: Optional[List]=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        if smpl_params is not None:
            smpl_params = smpl_params.clone().detach().cpu()
            pandas.to_pickle(smpl_params, os.path.join(point_cloud_path, "smpl_params.pkl"))

        if deformer is not None:
            deformer.dump_trans_grid(Path(point_cloud_path))

        # save person infos
        if people_infos is not None:
            for people_info in people_infos:
                # Though it's dimension could be different, we should handle that during data-loading time.
                smpl_poses = torch.cat([
                    people_info.smpl_scale.reshape(1,1).repeat(len(people_info.smpl_global_poses), 1),
                    people_info.smpl_global_poses,
                    people_info.smpl_local_poses
                ], dim=-1)
                people_info.human_scene.save(iteration, smpl_params=smpl_poses, deformer=people_info.smpl_deformer)  
                
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale] if len(self.train_cameras) > 0 else self.train_cameras

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale] if len(self.test_cameras) > 0 else self.test_cameras
    
    
        
class HumanScene(Scene):
    def __init__(
        self, 
        args: ModelParams,
        human_path: Path, 
        model_path: Path, 
        load_path: Path,
        eval: bool=False, 
        sh_degree: int=0, 
        scene_fnames=[], 
        mask_path: str= "", 
        load_iteration=None, 
        resolution_scales=[1.0], 
        init_opacity: float=0.9, 
        data_resolution=1,
        ):
        self.init_w_normal = True
        self.gaussians = GaussianModel(sh_degree)
        self.loaded_iter = None
        self.model_path = model_path
        self.load_path = load_path
        pnum = model_path.name

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(str(self.load_path))
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained human model {pnum} at iteration {self.loaded_iter}")

        self.train_cameras = {}
        self.test_cameras = {}
        human_scene_info = sceneLoadTypeCallbacks["Human"](
            human_path, 
            eval, 
            scene_fnames=scene_fnames, 
            person_mask_path=mask_path,
            )
        self.beta = human_scene_info.train_cameras[0].smpl_param[0, -10:].astype(np.float32) # first frame SMPL beta

        # setting for define gaussian scale (which is critical for gaussin radius)        
        self.cameras_extent = human_scene_info.nerf_normalization["radius"]

        # make cameras
        data_device = "cpu"     # for faster train, u can set it as 'cuda', but would be super heavy
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            if False: # For fast-loading options (not for optimizations, option for debugging)
                self.train_cameras = scene_info.train_cameras
            else:
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                                    human_scene_info.train_cameras, 
                                    resolution_scale, 
                                    args,
                                    data_resolution, 
                                    data_device, 
                                    )
            if len(human_scene_info.test_cameras) > 0:
                print("Loading Test Cameras")
            else:
                print("No Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                                    human_scene_info.test_cameras, 
                                    resolution_scale, 
                                    args,
                                    data_resolution, 
                                    data_device, 
                                    )

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(str(self.load_path), f"iteration_{self.loaded_iter}", f"actor_{pnum}.ply"))
            print(f"Loaded Optimized Point Clouds from {str(self.load_path)}")
        else:
            self.gaussians.create_from_pcd(human_scene_info.point_cloud, self.cameras_extent, self.init_w_normal, init_opacity)
            print("Loaded Initial Point Clouds")