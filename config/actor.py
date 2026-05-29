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


class HumanOptimizationParams:
    def __init__(self):
        # learning rate
        self.position_lr_init = 0.001           # 0.00016
        self.position_lr_final = 0.000002        # 0.00 00 016
        self.position_lr_delay_mult = 0.02      # 0.01
        self.position_lr_max_steps = 6000       # 30_000
        self.feature_lr = 0.0025                  # 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005                 # 0.005 (original)
        self.rotation_lr = 0.001
        # weights
        self.percent_dense = 0.01                # 0.01   if low : less clone more split. 
        self.lambda_dssim = 0.2
        self.lambda_mask = 0.01
        self.lambda_trans_reg = 0.1    
        # iteration settings
        self.densification_interval = 500 # NOTE 500        # originally: 100 
        self.opacity_reset_interval = 1500       # (do reset on time (after double densification))
        self.densify_from_iter = 500 # NOTE 500
        self.densify_until_iter = 5000 # NOTE 15_000
        # hyperparameters
        self.clip_init_smpl_opacity = False
        self.smpl_opacity_clip_min = 0.9 # x1/4 then original method 
        self.densify_grad_threshold = 0.00005   # x1/4 then original method          

        # ------------------------------ SMPL related fitting options ------------------------------
        self.do_smpl_mod = True
        self.fix_init_smpls_verts = False            #  If True, THE GS corresponding to the initial SMPL vertices does not be pruned and instead, returns initial points
        self.track_gs_parent_id = True              # We always need to track it's parent

        self.allow_init_smpl_splitting = True
        self.allow_init_smpl_cloning = True
        self.allow_init_smpl_pruning = False

        self.split_sharp_gaussian = False # regularizing Long (sharp) gaussian
        
        # ------------------------------ Human Gaussian settings ------------------------------
        self.sh_degree = 2
        self.view_dir_reg = False

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
        # paths & main options
        self._source_path = ""
        self._model_path = ""
        self._background_path = ""
        self.mask_path = ""
        self.foreground_mask_path = "none"
        self.textual_inversion_path = ""
        self.textual_inversion_method = ""
        self._images = "images"
        self.data_device = "cuda"
        self.sh_degree = 3                                      # SH degree for scene
        self._resolution = -1
        
        # bool options
        self.eval = False
        self.eval_with_black_bg = False
        self.use_canon = False                                  
        self.use_canon_single_camera = False                    # Valid only if it's training canonical (use single forward camera)
        self._white_background = False
        self.random_background = False
        self._opt_smpl = False
        self.use_trans_grid = False
        self.smpl_view_dir_reg = True
        self.clip_init_smpl_opacity = False                             # clip SMPL init opacity 
        
        self.use_adaptive_rgb_loss = False                      # use adpative loss based on MAX noise range of diffusion
        self.use_lpips_loss = False
        self.use_density_reg_loss = False                       # apply density regularize loss or not. (in DGM rendering)
        self.use_novel_view_density_reg = False                 # If True, calculate novel_view density_reg (Should not use together with use_diffusion_guidance)
        self.use_mask_loss = False                                   # If True, get mask loss  
        
        # hyperparams
        self.iter_clip_person_shs = 10                            # clip shs every n iters
        self.smpl_opacity_clip_min = 0.7
        self.human_sh_degree = 2
        self.human_sh_degree_enlarge_interval = 2000        # prev: 1500
        self.dilate_ratio = 0.00

        # ------------------------------ SMPL optimization ------------------------------
        self.iter_fix_smpl_init_verts = 1500 # NOTE 1500                        # from this iter, SMPL vertices move freely
        self.iter_smpl_densify = 1500 # NOTE 1500                              # Option, that turning on densify & splitting of human gaussians
        self.iter_densify_smpl_until = 3500                        # Do densify until this iterations (# of densify is different for each people)
        self.person_smpl_reset = -200000                        # do reset right after second densification. (just skip resetting here)
        self.iter_prune_smpl_until = 9000 # NOTE 7000                       # Prune invalid points until 7000 iterations

        # ------------------------------ Refinement ------------------------------
        self.start_deform = 2000 # NOTE 2000

        # ------------------------------ Diffusion ------------------------------
        # opt settings
        self.use_diffusion_guidance = False
        self.apply_dgm_every_n = 1 # Apply Diffusion Guidance every n iter
        self.dgm_start_iter = 1000 # NOTE 1000

        # noise & camera settings
        self.dgm_noise_sched = "time_annealing"
        self.dgm_random_sample = True # If False, only upper bound is defined
        self.dgm_camera_sched = "default" # camera sampling strategy
        self.dgm_cfg_sched = "default"

        # Textual Inversion Settings
        self.use_ti_in_controlnet = False
        self.use_ti_free_prompt_on_controlnet = False
        self.ti_chkpt_epoch = -1 # -1 means loading most recent checkpoints

        # diffusion settings
        self.dgm_use_cfg_rescale = True                     
        self.dgm_hard_masking = False
        self.dgm_use_optimizer_masking = False                  # Instead of masking with rendered visibility, just mask optimizers
        self.dgm_guidance_decay = False        
        self.dgm_enable_sdop = True
        self.dgm_use_aux_prompt = True
        self.dgm_use_view_prompt = True
        
        self.dgm_cfg_scale = 100.                                # cfg scale
        self.dgm_controlnet_weight = 0.7                         # 1.0 default. for inpainting, 0.5 is recommended. for img2img, 0.7 is recommended.
        self.dgm_minimum_mask_thrs = 0.02
        self.dgm_cfg_rescale_weight = 0.8
        self.dgm_inpaint_guidance_scale = 7.5
        self.dgm_num_inference_steps = 20
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.view_dir_reg = False
        self.disc_gaussian = False
        self.log_interval = 100
        self.render_type = "wave" # gt / wave / circle / fix
        self.use_wandb = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # iteration settings
        self.iterations = 15_000
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500 # NOTE 500
        self.densify_until_iter = 5000 # NOTE 15_000
        self.densify_grad_threshold = 0.0002
        self.deform_lr_max_steps = 40_000
        # learning rate
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 10_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        # weights
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_mask = 0.01      
        self.lambda_lpips = 0.1
        self.lambda_density_reg = 0.1
        self.lambda_trans_reg = 10
        self.lambda_init_smpl_verts_reg = 1e6
        self.lambda_rgb_loss = 1e4
        self.lambda_opacity_reg = 1e-5 # 1e-3
        self.lambda_color_reg = 1e-5 # 1e-3
        # diffusion guidance
        self.lambda_dg_loss = 1. # diffusion guidance
        self.lambda_dgm_percep = 1.0
        self.lambda_dgm_rgb = 0.1
        self.lambda_cd_loss = 0 # custom diffusion
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
