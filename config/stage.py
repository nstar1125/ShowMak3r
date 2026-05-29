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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    if value:
                        group.add_argument("--no_" + key, default=value, action="store_false")
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()

        skip_lists = []
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                if arg[0] in skip_lists:
                    continue
                setattr(group, arg[0], arg[1])
            elif arg[0][:3] == "no_" and arg[0][3:] in vars(self):
                if not arg[1]:
                    print(f"turning off {arg[0][3:]} {arg[1]}")
                    setattr(group, arg[0][3:], False)
                    skip_lists.append(arg[0][3:])
        return group


class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3                                      # SH degree for scene
        self._source_path = ""
        self._background_path = ""
        self._model_path = ""        
        self.data_device = "cuda"
        self.eval = False
        self._resolution = -1
        self.dilate_ratio = 0.02
        self._white_background = True
        self.random_background = False
        super().__init__(parser, "Loading Parameters", sentinel)

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.view_dir_reg = False
        self.disc_gaussian = False
        # Render
        self.render_interval = 100
        # Log
        self.use_wandb = False
        self.wandb_interval = 100
        self.wandb_project = "ShowMak3r_stage"
        self.wandb_name = None
        self.wandb_resume = "allow"
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # Iterations
        self.iterations = 15_000
        self.opacity_reset_interval = 3000
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.exclude_frames_until_iter = 7000
        self.end_depth = 5000
        # Learning Rate
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 10_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        # Other Hyperparameters
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        # self.lambda_lpips = 0.1
        self.densify_grad_threshold = 0.0002
        self.mono_depth_lambda = 0.2
        self.smooth_loss_lambda = 0.5
        self.depth_tolerance = 0.1 # 5.0

        super().__init__(parser, "Optimization Parameters")