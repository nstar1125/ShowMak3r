
import importlib
numpy = importlib.import_module('numpy')
numpy.float = numpy.float32
numpy.int = numpy.int32
numpy.bool = numpy.bool_
numpy.unicode = numpy.unicode_
numpy.complex = numpy.complex_
numpy.object = numpy.object_
numpy.str = numpy.dtype.str

import os
import json
import sys
import cv2
from typing import List, Union, NamedTuple, Any, Optional, Dict
from random import randint, random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import shutil
import torch
import pandas
from tqdm import tqdm, trange
from omegaconf import OmegaConf

from config.actor import ModelParams, PipelineParams, OptimizationParams, HumanOptimizationParams
from showmak3r.pipeline.dataset.composite_loader import load_composite_data
from showmak3r.pipeline.refine.deform_branch import DeformModel, get_residuals
from showmak3r.pipeline.renderer.renderer_wrapper import render_full_video, render_full_canonical, \
    render_insertion_video, render_novel_pose, render_dynamic_frame

from showmak3r.utils.general_utils import safe_state
from showmak3r.vis.viser_composite import visualize_4dgs


def testing(
        dataset, 
        opt, 
        pipe, 
        checkpoint, 
        exp_name='default', 
        ):
    # Human train settings
    human_train_opt = HumanOptimizationParams()
    human_train_opt.view_dir_reg = dataset.smpl_view_dir_reg
    human_train_opt.sh_degree = dataset.human_sh_degree
    
    # Load test datasets
    scene, _, people_infos = \
        load_composite_data(
            dataset=dataset,
            pipe=pipe,
            type="test", # {ti, train, test}
            iteration=-1,
            exp_name=exp_name,
            human_train_opt=human_train_opt,
        )
    iteration = people_infos[0].human_scene.loaded_iter
    
    # Load face-fitting model
    if dataset.use_deform:
        deform_pipe = DeformModel()
        deform_pipe.load_weights(output_path=Path(scene.model_path) / f"iteration_{iteration}")
    else:
        deform_pipe = None

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    black_bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    eval_bg = black_bg if dataset.eval_with_black_bg else white_bg

    # ----------------------------------- render videos -----------------------------------
    if dataset.render:
        with torch.no_grad():
            os.makedirs(scene.model_path + "/testing", exist_ok=True)
            # # 1. render ground-truth and novel-view videos
            # print("[INFO] Start rendering ground truth video")
            # render_full_video(Path(scene.model_path)/ "testing", scene, people_infos, 
            #                         pipe, background, deform_pipe, -1, dataset.use_deform, iteration, type="gt")
            
            print("[INFO] Start rendering novel view video")
            render_full_video(Path(scene.model_path)/ "testing", scene, people_infos, 
                                pipe, background, deform_pipe, -1, dataset.use_deform, iteration, type="fix") # circle / wave / fix

            # # 2. human deletion
            # print("[INFO] Start rendering actor deletion video")
            # render_full_video(Path(scene.model_path)/ "testing", scene, people_infos, 
            #                         pipe, background, deform_pipe, -1, dataset.use_deform, iteration, type="gt", delete_pid=1)
            
            # # 3. human relocation
            # print("[INFO] Start rendering actor relocation video")
            # render_full_video(Path(scene.model_path)/ "testing", scene, people_infos, 
            #                         pipe, background, deform_pipe, -1, dataset.use_deform, iteration, type="gt", offset_pid=1, offset_xyz=[2.0, -0.5, 5.0]) 

            # # 4. dynamic frame
            # print("[INFO] Start rendering dynamic frame")
            # render_dynamic_frame(
            #     Path(scene.model_path)/ "testing", scene, people_infos, pipe,
            #     background, deform_pipe, -1, dataset.use_deform, iteration, cam_id=30
            # )

            # # 5. human insertion
            # insert_path = "./results/TBBT/TBBT_test/exp15" # ./results/<project_name>/<exp_name> (trained without refinement)
            # if Path(insert_path).exists():
            #     print("[INFO] Start rendering actor insertion video")
            #     render_insertion_video(Path(scene.model_path)/ "testing", 
            #                     insert_path, 
            #                     scene, people_infos, dataset, pipe, background, 
            #                     deform_pipe, -1, dataset.use_deform, 
            #                     iteration, insert_offset_xyz=[0.5, 0.2, -3.0])

            # # 6. pose manipulate
            # POSE_FILE_NAME="./config/data/animation/aist_demo.npz"
            # demo_poses_np = np.load(POSE_FILE_NAME, allow_pickle=True)["poses"]     # poses (320, 72)
            # demo_poses = torch.from_numpy(demo_poses_np).float().cuda()
            # print("[INFO] Start rendering pose manipulation video")
            # render_novel_pose(
            #     Path(scene.model_path)/ "testing", scene, demo_poses, people_infos, pipe, background, 
            #     deform_pipe, -1, dataset.use_deform, iteration, target_pid=1, offset_xyz=[0.1,-0.6,1.75])

            
    # ----------------------------------- realtime visualization -----------------------------------
    if dataset.viser:
        visualize_4dgs(
            scene,
            people_infos,
            pipe,
            background,
            deform_pipe,
            share=False,
            port=8080,
        )

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])  
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--exp_name", type=str, default ="debug")
    parser.add_argument("--use_deform", action='store_true')
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--viser", action='store_true')

    args = parser.parse_args(sys.argv[1:])

    lp_extracted = lp.extract(args)
    lp_extracted.eval = True
    lp_extracted.use_deform = args.use_deform
    lp_extracted.render = args.render
    lp_extracted.viser = args.viser

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    testing(
            lp_extracted, 
            op.extract(args), 
            pp.extract(args), 
            args.start_checkpoint, 
            args.exp_name, 
        )

    # All done
    print("\nTesting complete.")
