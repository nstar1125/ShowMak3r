import os
import time
from pathlib import Path
from typing import List, Union, NamedTuple, Any, Optional, Dict

import torch
import pandas
import numpy as np
import cv2
from tqdm import tqdm, trange

from config.actor import ModelParams, PipelineParams
from showmak3r.pipeline.scene import Scene, GaussianModel, HumanScene
from showmak3r.pipeline.smpl_deform.deformer import SMPLDeformer
from showmak3r.utils.draw_op_jnts import smpl_joints2op_joints
from showmak3r.utils.graphics_utils import project_points_to_cam
from showmak3r.utils.system_utils import searchForMaxIteration
from showmak3r.pipeline.diffusion.guidance.joint_utils import filter_invisible_face_joints_w_prompts, get_view_prompt_of_body, combine_prompts



class PersonTrain(NamedTuple):
    uids : Union[List, torch.Tensor]        # for trans_grids
    fnames : Union[List, torch.Tensor]      # to check whether it's a scene requiring human
    model_path: Path
    smpl_local_poses: torch.Tensor
    smpl_global_poses: torch.Tensor
    detected_bbox: List
    local_pose_optimizer: Optional[torch.optim.Optimizer]
    global_pose_optimizer: Optional[torch.optim.Optimizer]
    smpl_scale_optimizer: Optional[torch.optim.Optimizer]
    beta: torch.Tensor
    smpl_deformer: SMPLDeformer
    gaussians: GaussianModel
    do_trans_grid: bool
    trans_grids: Optional[torch.Tensor]
    grid_optimizer: Optional[torch.optim.Optimizer]
    view_dir_reg: bool
    human_scene: Scene
    person_number: str
    smpl_scale: torch.Tensor
    init_smpl_jnts: torch.Tensor
    cam_centers: torch.Tensor
    cc_smpl_dir: torch.Tensor    
    representative_img: torch.Tensor
    misc: Dict   


def load_composite_data(
    dataset : ModelParams, 
    pipe : PipelineParams, 
    type: str="train",
    iteration : int=-1, 
    exp_name: str="debug", 
    human_train_opt = None,
    **kwargs
    ):
    '''
    Load stage gaussians and processed actors information 
    
    Returns:
        scene: Scene object of the stage (stage gaussians, train cameras)
        people_infos: PersonTrain objects of each actor (SMPL parameters)
    '''

    # ------------------------- load stage scene -------------------------
    scene_gaussians = GaussianModel(dataset.sh_degree)
    print(f"[INFO] Loading scene data")
    if type == "ti": # train_ti.py
        scene = Scene(dataset, scene_gaussians, scene_type="actor", load_iteration=iteration, view_dir_reg=pipe.view_dir_reg, exp_name=None)
    elif type == "train": # train_actor.py
        scene = Scene(dataset, scene_gaussians, scene_type="actor", load_iteration=iteration, view_dir_reg=pipe.view_dir_reg, exp_name=None)
        # make output path
        scene.model_path = str(os.path.join(str(scene.model_path), exp_name))
        os.makedirs(scene.model_path, exist_ok=True)
    elif type == "test":
        scene = Scene(dataset, scene_gaussians, scene_type="actor", load_iteration=iteration, view_dir_reg=pipe.view_dir_reg, exp_name=None)
        scene.model_path = str(os.path.join(str(scene.model_path), exp_name))
        loaded_iter = searchForMaxIteration(Path(scene.model_path))
        assert os.path.exists(scene.model_path), f"[ERROR] {scene.model_path} does not exist"
    else:
        raise AssertionError("[ERROR] Invalid type name")

    # get scene cameras
    scene_train_cams = scene.getTrainCameras()
    scene_test_cams = scene.getTestCameras()
    scene_cameras = scene_train_cams + scene_test_cams
    scene_cameras = sorted(scene_cameras.copy(), key=lambda x: x.fname)
    
    scene_cam_dict = dict()
    for cam in scene_cameras:
        fname = cam.fname
        scene_cam_dict[fname] = cam
    scene_fname_list = sorted(list(scene_cam_dict.keys()))

    # Load foreground masks
    if dataset.foreground_mask_path != "" and dataset.foreground_mask_path != "none":
        foreground_mask_path = Path(dataset.foreground_mask_path)
        fmask_dict = dict()
        for fmask_path in sorted(list(foreground_mask_path.glob("*.png"))+list(foreground_mask_path.glob("*.jpg"))):
            fname = fmask_path.name.split(".")[0]
            fmask_dict[fname] = torch.from_numpy(cv2.imread(str(fmask_path), 0)>1).float().squeeze()
        scene.fmask_dict = fmask_dict
    else:
        foreground_mask_path = ""
        scene.fmask_dict = None
    
    # ------------------------- load human datasets -------------------------
    # set hyperparameters
    human_sh_degree = human_train_opt.sh_degree
    human_view_dir_reg = human_train_opt.view_dir_reg

    # get actor directories
    actor_path = Path(dataset.model_path) / "actor"
    human_model_dict = dict() # pnum: actor_dir (ex. 001, 002, ...)
    pnum_list = sorted([p for p in os.listdir(actor_path) if p.isdigit()]) # only directories
    for pnum in pnum_list:
        human_model_dict[pnum] = actor_path / pnum
    
    # get trained actor path
    if type == "test":
        human_load_path = Path(dataset.model_path) / exp_name
    else:
        human_load_path = None
    
    # set all smpls as neutral
    smpl_genders = ['neutral' for _ in range(len(human_model_dict))]

    # start loading human datas
    people_infos = []
    for i, pnum in enumerate(pnum_list):
        print(f"[INFO] Loading human {pnum} data")
        human_model_path = human_model_dict[pnum]
        human_data_path = human_model_path / "optimized.pkl" # contains SMPL + cameras
        human_mask_dir = human_model_path / 'masks' # personal masks
        if not Path(human_mask_dir).exists():
            human_mask_dir = ""
        
        # turn eval for testing
        if type == "test":
            _loaded_iter = loaded_iter
            _eval = True
        else:
            _loaded_iter = None
            _eval = False
            
        # load human scene object
        human_scene = HumanScene(
                            dataset,
                            human_data_path, 
                            human_model_path, 
                            human_load_path,
                            eval=_eval, 
                            sh_degree=human_sh_degree, 
                            scene_fnames=scene_fname_list, 
                            mask_path=human_mask_dir,
                            data_resolution=dataset.resolution,
                            load_iteration=_loaded_iter,
                            )

        person_gaussians = human_scene.gaussians
        if type == "train":
            person_gaussians.training_setup(human_train_opt)

        # cameras in human_scene contains only smpls, not images
        train_cameras = human_scene.getTrainCameras()
        test_cameras = human_scene.getTestCameras()
        cameras = train_cameras + test_cameras
        cameras = sorted(cameras.copy(), key=lambda x: x.fname)
        
        uids = [cam.uid for cam in cameras]
        fname_list = [cam.fname for cam in cameras]
        
        # extract smpl poses
        smpl_local_poses = []
        smpl_global_poses = []
        smpl_params = []
        bbox_list = []
        cam_centers = []
        
        largest_res = -1
        largest_img = None
        for cam in cameras:
            smpl_local_poses.append(cam.smpl_param[:,4:76])
            smpl_global_poses.append(cam.smpl_param[:,1:4])
            smpl_params.append(cam.smpl_param.clone().detach().float())
            bbox_list.append(cam.bbox is not None)

            cam_center = scene_cam_dict[cam.fname].camera_center.clone().detach().float().squeeze()
            cam_centers.append(cam_center)

            if max(cam.gt_image.shape) > largest_res:
                largest_img = cam.gt_image
                largest_res = max(cam.gt_image.shape)
            
        # smpl batch
        smpl_local_poses = torch.cat(smpl_local_poses, dim=0)
        smpl_local_poses = smpl_local_poses.float().cuda()

        smpl_global_poses = torch.cat(smpl_global_poses, dim=0)
        smpl_global_poses = smpl_global_poses.float().cuda()

        init_smpl_params = torch.cat(smpl_params, dim=0).cuda()
        smpl_scale = init_smpl_params[:, 0].mean().detach()

        # set optimizers and camera centers for training
        if type == "train":
            smpl_local_poses = smpl_local_poses.requires_grad_()
            smpl_global_poses = smpl_global_poses.requires_grad_()
            smpl_scale = smpl_scale.requires_grad_()

            # smpl optimizers
            local_pose_optimizer = torch.optim.Adam([smpl_local_poses], lr=1e-4)
            global_pose_optimizer = torch.optim.Adam([smpl_global_poses], lr=1e-4)
            smpl_scale_optimizer = torch.optim.Adam([smpl_scale], lr=1e-4)
            smpl_scale = smpl_scale.reshape(-1)

            # calcuate camera centers and directions
            cam_centers = torch.stack(cam_centers, dim=0)
            cc_smpl_dir = smpl_global_poses.clone().detach() - cam_centers
        else:
            local_pose_optimizer = None
            global_pose_optimizer = None
            smpl_scale_optimizer = None
            cam_centers = None
            cc_smpl_dir = None

        # define smpl deformer
        smpl_gender = smpl_genders[i]
        beta = torch.from_numpy(human_scene.beta).float().cuda()
        smpl_canon_scale = 1.0
        smpl_deformer = SMPLDeformer(gender=smpl_gender, beta=beta, smpl_scale=smpl_canon_scale)

        # extract SMPL joints and vertices in canonical space
        smpl_output = smpl_deformer.smpl_server(init_smpl_params)
        smpl_jnts = smpl_output['smpl_jnts'].detach().cpu()
        smpl_verts = smpl_output['smpl_verts'].detach().cpu()

        # initialize transform grid, if needed
        if type == "train":
            do_trans_grid = dataset.use_trans_grid
            if do_trans_grid:
                n_frames = len(human_scene.getTrainCameras())
                trans_grids, grid_optimizer = smpl_deformer.activate_trans_grid(n_frames=n_frames)

                if trans_grids is not None:
                    print(f"[INFO] trans grids activated, n_frames: {n_frames}")
                else:
                    print(f"[INFO] trans grids failed to be initalized. invalid n_frames: {n_frames}")
                    do_trans_grid = False
            else:
                trans_grids = None
                grid_optimizer = None
        else:
            do_trans_grid = False
            trans_grids = None
            grid_optimizer = None
        
        # misc dict
        if type == "train":
            misc_dict = dict(
                optimized_step=0
            )
        else:
            misc_dict = dict()

        people_infos.append(
                PersonTrain(
                    uids = uids,
                    fnames = fname_list,
                    smpl_local_poses = smpl_local_poses,
                    smpl_scale = smpl_scale,
                    smpl_global_poses = smpl_global_poses,
                    detected_bbox = bbox_list,
                    local_pose_optimizer = local_pose_optimizer,
                    global_pose_optimizer = global_pose_optimizer,
                    smpl_scale_optimizer = smpl_scale_optimizer,
                    model_path = human_model_path,
                    beta = beta,
                    smpl_deformer = smpl_deformer,
                    gaussians = person_gaussians,
                    do_trans_grid = do_trans_grid,
                    trans_grids = trans_grids,
                    grid_optimizer = grid_optimizer,
                    view_dir_reg = human_view_dir_reg,
                    human_scene = human_scene,
                    person_number=pnum,
                    init_smpl_jnts = smpl_jnts,
                    cam_centers = cam_centers,
                    cc_smpl_dir = cc_smpl_dir,
                    representative_img = largest_img,
                    misc = misc_dict
                )
            )

        if type == "ti":
            # generate prompts from smpl joints
            project_op_jnts = []
            op_3d_jnts = []
            new_prompts = []
            for i, fname in enumerate(fname_list):
                scene_cam = scene_cam_dict[fname]
                
                # get 3d joints
                smpl_jnt = smpl_jnts[i].clone().detach().cpu()
                pj_jnts = project_points_to_cam(scene_cam, smpl_jnt.squeeze().numpy(), image_res=None)
                op_joints = smpl_joints2op_joints(pj_jnts)
                op_3d_jnt = smpl_joints2op_joints(smpl_jnt.squeeze().numpy())

                # get prompts from 3d joints
                lower_body_prompt = get_view_prompt_of_body(op_3d_jnt, scene_cam, is_lower_body=True)
                upper_body_prompt = get_view_prompt_of_body(op_3d_jnt, scene_cam, is_lower_body=False)
                filtered_op_3d_jnt, head_prompt = filter_invisible_face_joints_w_prompts(op_3d_jnt, scene_cam)
                image_res = (scene_cam.image_height, scene_cam.image_width)
                
                # combine prompts
                new_prompt = combine_prompts(head_prompt, upper_body_prompt, lower_body_prompt, op_joints, image_res)
                
                project_op_jnts.append(op_joints)
                op_3d_jnts.append(op_3d_jnt)
                new_prompts.append(new_prompt)
                
            # append prompt informations
            people_infos[-1].misc['projected_op_jnts'] = project_op_jnts
            people_infos[-1].misc['3d_op_jnts'] = op_3d_jnts
            people_infos[-1].misc['body_prompts'] = new_prompts
        elif type == "train":
            # append initial vertices
            people_infos[-1].misc['smpl_verts'] = smpl_verts
    
    print(f"[INFO] Loaded scene data. {len(people_infos)} actors found.")
    
    return scene, scene_gaussians, people_infos