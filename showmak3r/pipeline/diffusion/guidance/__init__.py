import os
import time
import random
import PIL
import copy
import math
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage import exposure
from typing import List, Dict, Optional

from showmak3r.utils.image_utils import img_add_text
from showmak3r.utils.jnts_utils import extract_square_bbox
from showmak3r.utils.camera_utils import focal2fov, fov2focal, project_points_to_cam, getProjectionMatrix

from showmak3r.pipeline.diffusion.guidance.diffusion_module.controlnet_guidance import OPControlNet
from showmak3r.pipeline.diffusion.guidance.cam_utils import MiniCam, orbit_camera, OrbitCamera
from showmak3r.pipeline.diffusion.guidance.color_correction import compute_cc_target, apply_color_correction

class DiffusionGuidance:
    '''
    DGM(Diffusion Guidance Module) 
    
    ControlNet diffusion guidance from rendered images.
    '''
    def __init__(
                self, 
                opt, 
                log_dir: Path, 
                textual_inversion_path: Optional[Path]=None, 
                textual_inversion_in_controlnet: bool = False,
                use_ti_free_prompt_on_controlnet: bool = False,
                guidance_scale: float=100,
                inpaint_guidance_scale: float=7.5,
                controlnet_weight: float=1.0,
                lambda_percep: float=0.0,
                lambda_rgb: float=0.1,
                random_noise_step: bool = True,
                noise_sched: str = "time_annealing",
                camera_sched: str = "default",
                do_guid_sched: bool=False,
                sd_version: str="1.5",
                use_aux_prompt: bool=False,
                use_view_prompt: bool=False,
                cfg_sched: str="default",
                ti_load_epoch: int=-1,
                ):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.       
        self.sd_version = sd_version
        self.ti_load_epoch = ti_load_epoch
        self.ti_in_controlnet = textual_inversion_in_controlnet
        self.use_ti_free_prompt_on_controlnet = use_ti_free_prompt_on_controlnet

        if self.sd_version in ["2.1", "2.0"]:
            self.max_res = 768
        else:
            self.max_res = 512

        # ---------------------------------- Noise scheduling related ----------------------------------
        self.annealing_noise_decay = False
        self.current_stage = 0
        self.noise_sched = noise_sched
        # noise schedule related settings
        if noise_sched == "time_annealing_woreset":
            # (It was default option, before 11.7 19:00 )
            # Based on HiFA
            self.annealing_noise_decay = True
            self.random_noise_step = random_noise_step
            self.noise_stage_iters = [2000]
            self.noise_minmax = [
                [0.2, 0.98],
                [0.02, 0.2]
            ]
            self.noise_decay = [True, False]
            
        elif noise_sched == "time_annealing_1108":
            # Based on HiFA
            self.annealing_noise_decay = True
            self.random_noise_step = random_noise_step
            self.noise_stage_iters = [1000, 2000]
            self.noise_minmax = [
                [0.5, 0.98],
                [0.2, 0.98],
                [0.02, 0.2]
            ]
            self.noise_decay = [False, True, False]
        
        elif noise_sched == "time_annealing": # default
            # Based on HiFA
            self.annealing_noise_decay = True
            self.random_noise_step = random_noise_step
            self.noise_stage_iters = [2000, 4000]
            self.noise_minmax = [
                [0.5, 0.98],
                [0.3, 0.98],
                [0.02, 0.3]
            ]
            self.noise_decay = [False, True, False]
            
        elif noise_sched == "time_annealing_ddim":
            # Based on HiFA
            self.annealing_noise_decay = True
            self.random_noise_step = random_noise_step
            self.noise_stage_iters = [2000, 4000]       
            self.noise_minmax = [
                [0.5, 0.98],
                [0.2, 0.98],
                [0.2, 0.5]
            ]
            self.noise_decay = [False, True, False]

        elif noise_sched == "two_stage_set1":
            # based on Humannorm
            self.random_noise_step = True
            self.noise_stage_iters = [1000, 2000]       # (original: 5000, 8000
            self.noise_minmax = [
                [0.02, 0.8],
                [0.02, 0.8],
                [0.02, 0.2]
                ]
            self.noise_decay = [False, True, False]

        elif noise_sched == "two_stage_set2":
            # naive approach
            self.random_noise_step = True
            self.noise_stage_iters = [500, 2000, 4000]       # (original: 5000, 8000s
            self.noise_minmax = [
                [0.5, 0.98],
                [0.02, 0.98],
                [0.02, 0.5],
                [0.02, 0.2]
                ]
            self.noise_decay = [False, False, False, False]
        
        elif noise_sched == "two_stage_set3":
            # naive approach
            self.random_noise_step = True
            self.noise_stage_iters = [500, 1500, 2000]       # (original: 5000, 8000s
            self.noise_minmax = [
                [0.5, 0.98],
                [0.02, 0.8],
                [0.2, 0.5],
                [0.02, 0.2]
                ]
            self.noise_decay = [True, False, True, False]
        
        elif noise_sched == "1025_defacto":
            # old approach
            self.annealing_noise_decay = True
            self.random_noise_step = True
            self.noise_stage_iters = [2000]       # (original: 5000, 8000s
            self.noise_minmax = [
                [0.2, 0.98],
                [0.02, 0.2]
            ]
        
        else:
            raise NotImplementedError(f"[ERROR] Unexpected value '{self.noise_minmax}'")
        
        self.noise_stage_iters = [0, *self.noise_stage_iters]

        # ---------------------------------- cfg scheduling ----------------------------------
        if cfg_sched == 'default': # default
            self.use_cfg_sched = False
        elif cfg_sched == 'time_annealing':
            self.use_cfg_sched = True
            self.cfg_stage_iters = [2000, 4000]
            self.cfg_minmax = [
                [100, 100],
                [7.5, 100],
                [7.5, 7.5]
            ]
            self.cfg_decay = [False, True, False]
        else:
            raise NotImplementedError(f"[ERROR] Unexpected value '{self.noise_minmax}'")
            
        # ---------------------------------- Camera Scheduling Related ----------------------------------
        # probability
        self.camera_sched = camera_sched
        if camera_sched == 'default':
            self.use_zoomed_camera = False
            self.coarse_res_step = 1000

        elif camera_sched == 'canon_pose_mixed':
            self.use_zoomed_camera = True
            self.coarse_res_step = 1000
            self.camera_sched_iters = []
            self.camera_prob=[
                # stage 1
                {
                    'entire_body':0.7,
                    'canon_body':0.3
                }
            ]
        elif camera_sched == 'two_stage_zoom':
            # human norm style
            self.use_zoomed_camera = True
            self.coarse_res_step = 1000
            self.camera_sched_iters = [2000]
            self.camera_prob=[
                # stage 1
                {
                    'head':0.1, 
                    'upper_body':0.1, 
                    'lower_body':0.1,
                    'entire_body':0.4,
                    'canon_body':0.3
                },
                # stage 2
                {
                    'head':0.3, 
                    'upper_body':0.3, 
                    'lower_body':0.3,
                    'entire_body':0.05,
                    'canon_body':0.05
                }
            ]
            
        elif camera_sched == 'three_stage_zoom':
            # human norm style
            self.use_zoomed_camera = True
            self.coarse_res_step = 1000
            self.camera_sched_iters = [2500, 3500]
            self.camera_prob=[
                # stage 1
                {
                    'head':0.1, 
                    'upper_body':0.1, 
                    'lower_body':0.1,
                    'entire_body':0.4,
                    'canon_body':0.3
                },
                # stage 2
                {
                    'head':0.3, 
                    'upper_body':0.3, 
                    'lower_body':0.3,
                    'entire_body':0.1,
                    'canon_body':0.
                },
                # stage 3
                {
                    'head':0.2, 
                    'upper_body':0.2, 
                    'lower_body':0.2,
                    'entire_body':0.3,
                    'canon_body':0.1
                }
            ]

        elif camera_sched == 'defacto_old':
            # human norm style
            self.use_zoomed_camera = True
            self.coarse_res_step = 1000
            self.camera_sched_iters = [2000]
            self.camera_prob=[
                # stage 1
                {
                    'head':0.1, 
                    'upper_body':0.1, 
                    'lower_body':0.1,
                    'entire_body':0.4,
                    'canon_body':0.3
                },
                # stage 2
                {
                    'head':0.2, 
                    'upper_body':0.2, 
                    'lower_body':0.2,
                    'entire_body':0.3,
                    'canon_body':0.1
                }
            ]

        elif camera_sched == 'defacto': # default
            # Defacto during rebuttal
            # Increased the ratio of "head" to achieve better alignment
            # human norm style
            self.use_zoomed_camera = True
            self.coarse_res_step = 1000
            self.camera_sched_iters = [2000]
            self.camera_prob=[
                # stage 1
                {
                    'head':0.2, 
                    'upper_body':0.1, 
                    'lower_body':0.1,
                    'entire_body':0.4,
                    'canon_body':0.2
                },
                # stage 2
                {
                    'head':0.3, 
                    'upper_body':0.2, 
                    'lower_body':0.2,
                    'entire_body':0.2,
                    'canon_body':0.1
                }
            ]
            
            
        else:
            raise NotImplementedError(f"[ERROR] Unexpected camera_schedule method: '{self.camera_sched}'")

        if self.use_zoomed_camera:
            self.camera_sched_iters = [0, *self.camera_sched_iters]

        # --------------------------------------------------------------------------------
        
        # Some exp_settings
        self.adaptive_noise_sched = False
        self.color_correction = False       # It will be controlled with lambda outside
        self.use_aux_prompt = use_aux_prompt
        self.use_view_prompt = use_view_prompt
        self.controlnet_weight = controlnet_weight

        # Some weight settings
        self.lambda_rgb = lambda_rgb
        self.lambda_percep = lambda_percep
        
        assert (not (self.annealing_noise_decay and self.adaptive_noise_sched)), "invalid noise scheduling"
        if self.adaptive_noise_sched:
            raise NotImplementedError("Not tested yet")

        self.inpaint_guidance_scale = inpaint_guidance_scale
        self.guidance_scale = guidance_scale
        self.guidnace_sched_ratio = 1/4
        self.do_guid_sched = do_guid_sched
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        # self.guidance_sd = None
        # self.guidance_zero123 = None
        self.guidance_controlnet = None

        # self.enable_sd = False
        # self.enable_zero123 = False
        self.enable_controlnet = False

        # input text
        self.prompt = "a photo of a person"
        self.negative_prompt = "two people, low quality" 
        self.aux_pos_prompt = ", high quality, 8k uhd, realistic"
        self.aux_neg_prompt = ", lowres, bad anatomy, bad, hands, missing fingers, worst quality, nsfw"

        # training settings
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt

        # textual inversion setting
        self.textual_inversion_path = textual_inversion_path

        if textual_inversion_path is not None:
            self.prompt = "a photo of a <new1> person"

        # auxiliary prompt
        if self.use_aux_prompt:
            self.prompt = self.prompt + self.aux_pos_prompt
            self.negative_prompt = self.negative_prompt + self.aux_neg_prompt

        # save dir
        self.log_dir = log_dir
        self.diffusion_img_dir = self.log_dir
        
        self.log_dir.mkdir(exist_ok=True)
        self.diffusion_img_dir.mkdir(exist_ok=True)

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(
                    self, 
                    pnums: List, 
                    enable_controlnet: bool=True, 
                    do_cfg_rescale: bool=False,
                    ):
        self.input_img = None
        self.train_pnums = pnums
        self.step = dict()
        self.dg_lambda = dict()
        self.cd_lambda = dict()
        self.dg_loss_log_stack = dict()     # to store recent diffusion-guidance loss, to adaptively alter leraning rate & noise-level
        self.noise_max = dict()     # to store recent diffusion-guidance loss, to adaptively alter leraning rate & noise-level
        self.noise_min = dict()     # to store recent diffusion-guidance loss, to adaptively alter leraning rate & noise-level

        self.diffusion_img_dict = dict()
        self.input_img_dict = dict()
        self.diffusion_img_log_dir = dict()

        for pnum in pnums:
            self.step[pnum] = 0
            self.diffusion_img_dict[pnum] = []
            self.input_img_dict[pnum] = []

            self.diffusion_img_log_dir[pnum] = self.diffusion_img_dir / f"{pnum}"
            self.diffusion_img_log_dir[pnum].mkdir(exist_ok=True)
            self.dg_lambda[pnum] = 1.0
            self.cd_lambda[pnum] = self.opt.lambda_cd_init
            self.dg_loss_log_stack[pnum] = []

        self.enable_controlnet = enable_controlnet

        self.text_inv_dict = None
        if self.textual_inversion_path is not None:
            print("[INFO] start loading TI")
            self.text_inv_dict = self.load_textual_inversion_path_dict(pnums)

        if self.guidance_controlnet is None and self.enable_controlnet:
            print(f"[INFO] loading OpenposeControlNet...")
            self.guidance_controlnet = OPControlNet(
                                                device=self.device, 
                                                textual_inversion_path=self.text_inv_dict, 
                                                textual_inversion_in_controlnet=self.ti_in_controlnet,
                                                sd_version=self.sd_version,
                                                do_cfg_rescale=do_cfg_rescale,
                                                use_ti_free_prompt_on_controlnet=self.use_ti_free_prompt_on_controlnet
                                                )
            print(f"[INFO] loaded OpenposeControlNet!")

        # prepare embeddings
        with torch.no_grad():
            if self.enable_controlnet:
                self.guidance_controlnet.get_text_embeds([self.prompt], [self.negative_prompt])

        # set initial noise min-max here
        self.set_noise_level(self.noise_minmax[0])

    def load_textual_inversion_path_dict(self, pnums: List):
        if self.ti_load_epoch == -1:
            print(f"[INFO] Loading TI from {str(self.textual_inversion_path)}/**/pytorch_custom_diffusion_weights.bin")
            text_inv_list = Path(str(self.textual_inversion_path)).glob(f"**/pytorch_custom_diffusion_weights.bin")
        else:
            print(f"[INFO] Loading TI from {str(self.textual_inversion_path)}/{str(self.textual_inversion_expname)}/**/token_{self.ti_load_epoch}/pytorch_custom_diffusion_weights.bin")
            text_inv_list = Path(str(self.textual_inversion_path)).glob(f"{str(self.textual_inversion_expname)}/**/token_{self.ti_load_epoch}/pytorch_custom_diffusion_weights.bin")

        text_inv_dict = dict()
        for text_inv_path in tqdm(text_inv_list, desc="TI path lists"):
            if self.ti_load_epoch == -1:
                pnum = text_inv_path.parents[0].name
                if pnum[:5] == 'token':
                    continue
            else:
                pnum = text_inv_path.parents[1].name

            if pnum not in pnums:
                pnum = str(int(pnum))
            if not (pnum in pnums):
                print(f"[WARNING] skipped person number {pnum}, valid numbers: {pnums}")
                
            print(f"[INFO] Loading TI from {str(text_inv_path.parent)}")
            text_inv_dict[pnum] = text_inv_path.parent

        for k in text_inv_dict.keys():
            print(f"{k}: {str(text_inv_dict[k])}")
        
        return text_inv_dict

    def get_test_camera(self, smpl_param=None, smpl_deformer=None, n_cameras=18):
        """
        get mv cameras rendering canon / full-body / face / upper-body / lower-body
        """
        cam_dicts = dict()
        aux_prompt_dicts = dict()
        aux_neg_prompt_dicts = dict()
        new_smpl_dicts = dict()

        # 0. get base cameras (canon / entire)
        hors = range(-180, 180, 360 // n_cameras)
        ver = 0 
        for name in ['canon_body', 'full_body', 'head', 'upper_body', 'lower_body']:
            cam_dicts[name] = []
            aux_prompt_dicts[name] = []
            aux_neg_prompt_dicts[name] = []
            new_smpl_dicts[name] = []

        # prepare canon smpl params
        canon_smpl_param = smpl_param.clone().detach()
        canon_smpl_param[0, 7:-10] *= 0 # remove local rotation
        # set joint angle as canonical shape
        canon_smpl_param[0, 9] = torch.tensor((torch.pi / 6)).to(canon_smpl_param.device).float()
        canon_smpl_param[0, 12] = (-torch.tensor((torch.pi / 6))).to(canon_smpl_param.device).float()

        for hor in hors:
            pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius * 1.)
            cur_cam = MiniCam(
                pose,
                512,
                512,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )
            cam_dicts['canon_body'].append(cur_cam)
            cam_dicts['full_body'].append(cur_cam)
            
            # View prompting
            part_name = 'canon_body'
            if hor > -45 and hor < 45:
                prompt = self.prompt + f", front view, {part_name.split('_')[0]} {part_name.split('_')[1]}"
                negative_prompt = self.negative_prompt + ", back view"
                part_name += f"_front view {hor}"
            elif hor > 135 or hor < -135:
                prompt = self.prompt + f", back view, {part_name.split('_')[0]} {part_name.split('_')[1]}"
                negative_prompt = self.negative_prompt + ", front view"
                part_name += f"_back view {hor}"
            else:
                prompt = self.prompt + f", side view, {part_name.split('_')[0]} {part_name.split('_')[1]}"
                negative_prompt = self.negative_prompt
                part_name += f"_side view {hor}"

            aux_prompt_dicts['canon_body'].append(prompt)
            aux_neg_prompt_dicts['canon_body'].append(negative_prompt)
            new_smpl_dicts['canon_body'].append(canon_smpl_param.clone().detach())
            

            part_name = 'entire_body'
            if hor > -45 and hor < 45:
                prompt = self.prompt + f", front view, {part_name.split('_')[0]} {part_name.split('_')[1]}"
                negative_prompt = self.negative_prompt + ", back view"
                part_name += f"_front view {hor}"
            # elif hor > 135 or hor < -135: 
            elif hor > 115 or hor < -115:   # Updated (240108, wider back view range)
                prompt = self.prompt + f", back view, {part_name.split('_')[0]} {part_name.split('_')[1]}"
                negative_prompt = self.negative_prompt + ", front view"
                part_name += f"_back view {hor}"
            else:
                prompt = self.prompt + f", side view, {part_name.split('_')[0]} {part_name.split('_')[1]}"
                negative_prompt = self.negative_prompt
                part_name += f"_side view {hor}"
            aux_prompt_dicts['full_body'].append(prompt)
            aux_neg_prompt_dicts['full_body'].append(negative_prompt)
            new_smpl_dicts['full_body'].append(smpl_param.clone().detach())


            # 1. get zoomed cameras
            for part_name in ['head', 'upper_body', 'lower_body']:
                _cur_cam = copy.deepcopy(cur_cam)
                part_cam, part_smpl_param, head_prompts = self.get_zoomed_camera(_cur_cam, smpl_param.clone().detach(), smpl_deformer, part_name=part_name)

                cam_dicts[part_name].append(part_cam)
                new_smpl_dicts[part_name].append(part_smpl_param.clone().detach())

                if part_name == 'head':
                    prompt = head_prompts[0]
                    negative_prompt = head_prompts[1]
                else:
                    if hor > -45 and hor < 45:
                        prompt = self.prompt + f", front view, {part_name.split('_')[0]} {part_name.split('_')[1]}"
                        negative_prompt = self.negative_prompt + ", back view"
                        # part_name += f"_front view {hor}"
                    elif hor > 135 or hor < -135:
                        prompt = self.prompt + f", back view, {part_name.split('_')[0]} {part_name.split('_')[1]}"
                        negative_prompt = self.negative_prompt + ", front view"
                        # part_name += f"_back view {hor}"
                    else:
                        prompt = self.prompt + f", side view, {part_name.split('_')[0]} {part_name.split('_')[1]}"
                        negative_prompt = self.negative_prompt
                        # part_name += f"_side view {hor}"


                aux_prompt_dicts[part_name].append(prompt)
                aux_neg_prompt_dicts[part_name].append(negative_prompt)

        return cam_dicts, aux_prompt_dicts, aux_neg_prompt_dicts, new_smpl_dicts

    def get_render_camera(self, pnum, scale: float=1., get_single_fixed_camera: bool=False, smpl_param=None, smpl_deformer=None):
        """
        return a camera, that render a novel view
        """
        self.step[pnum] += 1

        # Select rendering resolution
        render_resolution = self.max_res//2 if self.step[pnum] < self.coarse_res_step else self.max_res

        get_zoomed_camera = False
        if self.use_zoomed_camera:
            assert (smpl_param is not None)
            assert (smpl_deformer is not None)

            cur_stage = self.get_cur_stage(self.step[pnum], is_noise=False)
            zoom_target_prob = self.camera_prob[cur_stage]
            assert sum(list(zoom_target_prob.values())) > 0.999, f"invalid prob set {zoom_target_prob}"

            # select parts first
            rand_value = random.random()
            _thrs = 0
            for k, v in zoom_target_prob.items():
                _thrs += v
                if rand_value <= _thrs:
                    part_name = k
                    break
            if part_name in ['entire_body', 'canon_body']:
                get_zoomed_camera = False
            else:
                get_zoomed_camera = True

            if part_name == 'canon_body':
                smpl_param[0, 7:-10] *= 0 # remove local rotation
                # set joint angle as canonical shape
                smpl_param[0, 9] = torch.tensor((torch.pi / 6)).to(smpl_param.device).float()
                smpl_param[0, 12] = (-torch.tensor((torch.pi / 6))).to(smpl_param.device).float()
                smpl_param[0, 9 + 3*12] = -torch.tensor((torch.pi / 6)).to(smpl_param.device).float()
                smpl_param[0, 9 + 3*13] = torch.tensor((torch.pi / 6)).to(smpl_param.device).float()
        else:
            part_name = "full_body"

        if part_name == "entire_body" or part_name == "canon_body":
            part_name = "full_body"


        # check whether we need to get zoomed or not.
        if get_zoomed_camera:
            radius = 0 
            # 0. set initial camera
            ver = 0
            hor = np.random.randint(-180, 180)
            pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius * scale * 1.3)
            init_cam = MiniCam(
                pose,
                render_resolution,
                render_resolution,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            cur_cam, smpl_param, _ = self.get_zoomed_camera(init_cam, smpl_param, smpl_deformer, part_name=part_name)
        else:
            min_ver = max(min(-30, -30 - self.opt.elevation), -70 - self.opt.elevation)
            max_ver = min(max(30, 30 - self.opt.elevation), 70 - self.opt.elevation)
            
            # render random view
            if get_single_fixed_camera:
                ver = 0
                hor = 0
            else:
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
            radius = 0
            pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius * scale)

            cur_cam = MiniCam(
                pose,
                render_resolution,
                render_resolution,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

        invert_bg_color = np.random.rand() > self.opt.invert_bg_prob

        if self.use_view_prompt and not part_name[:4]=='head':
            # check horizontal rotation values
            if hor > -45 and hor < 45:
                prompt = self.prompt + f", front view, {part_name.split('_')[0]} {part_name.split('_')[1]}"
                negative_prompt = self.negative_prompt + ", back view"
                part_name += f"_front view {hor}"
            elif hor > 135 or hor < -135:
                prompt = self.prompt + f", back view, {part_name.split('_')[0]} {part_name.split('_')[1]}"
                negative_prompt = self.negative_prompt + ", front view"
                part_name += f"_back view {hor}"
            else:
                prompt = self.prompt + f", side view, {part_name.split('_')[0]} {part_name.split('_')[1]}"
                negative_prompt = self.negative_prompt
                part_name += f"_side view {hor}"

            if self.enable_controlnet:
                self.guidance_controlnet.set_text_embeds([prompt], [negative_prompt])


        return cur_cam, invert_bg_color, [ver], [hor], [radius], smpl_param, part_name

    def get_zoomed_camera(self, init_cam, smpl_param, smpl_deformer, part_name='head'):
        """
        return a camera, that render a novel view
        """
        if part_name not in ['head', 'upper_body', 'lower_body']:
            raise NotImplementedError()

        if part_name == 'head':
            valid_inds = [15, 24, 25, 26, 27, 28]
            # make canonical 
            neck_joints = 12
            neck_backup = smpl_param[0, 4+3*neck_joints:4+3*neck_joints+3]
            head_joints = 15
            head_backup = smpl_param[0, 4+3*head_joints:4+3*head_joints+3]

            # reset smpl params
            smpl_param[0, 4:-10] *= 0
            smpl_param[0, 4+3*neck_joints:4+3*neck_joints+3] = neck_backup
            smpl_param[0, 4+3*head_joints:4+3*head_joints+3] = head_backup

            bbox_offset_ratio = 1.6
        elif part_name == 'upper_body':
            # turn off rotation of 0, 1, 2, 3 
            neutral_joints = [0,1,2,3]
            for j in neutral_joints:
                smpl_param[0, 4+3*j:4+3*j+3] *= 0

            valid_inds = [1,2, 12, 3,6,9,13,14,16,17,18,19,20,21,22,23]

            bbox_offset_ratio = 0.4
        elif part_name == 'lower_body':
            # turn off rotation of 0, 1
            neutral_joints = [0,1]
            for j in neutral_joints:
                smpl_param[0, 4+3*j:4+3*j+3] *= 0

            valid_inds = [1,2,4,5,7,8,10,11]
            bbox_offset_ratio = 0.4

        smpl_output = smpl_deformer.smpl_server(smpl_param)
        smpl_jnts = smpl_output['smpl_jnts'].detach().cpu()
        
        # get projected jnts
        pj_jnts = project_points_to_cam(init_cam, smpl_jnts.squeeze().numpy())
        valid_jnts = pj_jnts[valid_inds]

        if part_name == 'head' and self.use_view_prompt:
            left_eye = valid_jnts[-3]
            right_eye = valid_jnts[-4]
            left_ear = valid_jnts[-1]
            right_ear = valid_jnts[-2]

            if (left_eye[0] < right_eye[0]) or (left_ear[0] < right_ear[0]): 
                # side view case (located in sinlge ways)
                if left_eye[0] < right_ear[0] and left_eye[0] < left_ear[0]:
                    view = "side"
                elif left_eye[0] > right_ear[0] and left_eye[0] > left_ear[0]:
                    view = "side"
                elif right_eye[0] < right_ear[0] and right_eye[0] < left_ear[0]:
                    view = "side"
                elif right_eye[0] > right_ear[0] and right_eye[0] > left_ear[0]:
                    view = "side"
                else:
                    view = "back"
            else:
                # side view case (located in sinlge ways)
                if left_eye[0] < right_ear[0] and left_eye[0] < left_ear[0]:
                    view = "side"
                elif left_eye[0] > right_ear[0] and left_eye[0] > left_ear[0]:
                    view = "side"
                elif right_eye[0] < right_ear[0] and right_eye[0] < left_ear[0]:
                    view = "side"
                elif right_eye[0] > right_ear[0] and right_eye[0] > left_ear[0]:
                    view = "side"
                else:
                    view = "front"

            # modify text prompts\
            # check horizontal rotation values
            if view=='front':
                prompt = self.prompt + ", front view, head, face"
                negative_prompt = self.negative_prompt + ", back view"
                part_name += "_front view"
            elif view=='back':
                prompt = self.prompt + ", back view, head"
                negative_prompt = self.negative_prompt + ", front view"
                part_name += "_back view"
            else:
                prompt = self.prompt + ", side view, head"
                negative_prompt = self.negative_prompt
                part_name += "_side view"

            if self.enable_controlnet:
                self.guidance_controlnet.set_text_embeds([prompt], [negative_prompt])
        else:
            prompt = ""
            negative_prompt = ""
            
        
        bbox = extract_square_bbox(valid_jnts, offset_ratio=bbox_offset_ratio, get_square=True)

        # get crop info
        new_width = bbox[2] - bbox[0]
        org_width = init_cam.image_width
        org_height = init_cam.image_height
        center_offset = [(bbox[2] + bbox[0])/2 - org_width/2, (bbox[3] + bbox[1])/2 - org_height/2] 
        new_scale = new_width / org_width

        # extract crop focal length
        org_fx = fov2focal(init_cam.FoVx, org_width)
        org_fy = fov2focal(init_cam.FoVy, org_height)
        new_fx = org_fx / new_scale
        new_fy = org_fy / new_scale

        # 2. change fov, cx, cy according to here.
        new_fovx = focal2fov(new_fx, org_width)
        new_fovy = focal2fov(new_fy, org_height)
        new_cx = -center_offset[0] / (new_width/2)
        new_cy = -center_offset[1] / (new_width/2)

        init_cam.FoVx = new_fovx
        init_cam.FoVy = new_fovy
        init_cam.cx = new_cx
        init_cam.cy = new_cy      
        init_cam.projection_matrix = getProjectionMatrix(znear=init_cam.znear, zfar=init_cam.zfar, fovX=new_fovx, fovY=new_fovy, cx=new_cx, cy=new_cy).transpose(0,1).cuda()
        init_cam.full_proj_transform = (init_cam.world_view_transform.unsqueeze(0).bmm(init_cam.projection_matrix.unsqueeze(0))).squeeze(0)
        init_cam.camera_center = init_cam.world_view_transform.inverse()[3, :3]

        return init_cam, smpl_param, (prompt, negative_prompt)

    def get_lambda_cd(self, pnum):
        step_ratio = min(1, self.step[pnum] / self.opt.lambda_cd_end_step)
        self.cd_lambda[pnum] = (self.opt.lambda_cd_end - self.opt.lambda_cd_init) * step_ratio + self.opt.lambda_cd_init
        return self.cd_lambda[pnum]

    def set_noise_level(self, noise_level):
        if self.enable_controlnet:
            self.guidance_controlnet.set_noise_level(noise_level)

    def get_cur_stage(self, step, is_noise: bool=True, is_cfg: bool=False):
        if is_noise:
            if len(self.noise_stage_iters) == 1:
                return 0
            for i, thrs in enumerate(self.noise_stage_iters[1:]):
                if thrs > step:
                    return i
            return len(self.noise_stage_iters)-1
        elif is_cfg:
            if len(self.cfg_stage_iters) == 1:
                return 0
            for i, thrs in enumerate(self.cfg_stage_iters[1:]):
                if thrs > step:
                    return i
            return len(self.cfg_stage_iters)-1
        else:
            if len(self.camera_sched_iters) == 1:
                return 0
            for i, thrs in enumerate(self.camera_sched_iters[1:]):
                if thrs > step:
                    return i
            return len(self.camera_sched_iters)-1

    def get_noise_level(self):
        steps = torch.tensor(list(self.step.values())).float()
        avg_step = int(steps.mean().item())
        cur_stage = self.get_cur_stage(avg_step)
        if cur_stage < len(self.noise_stage_iters[1:]):
            local_step = avg_step - self.noise_stage_iters[cur_stage]
            local_step = local_step / (self.noise_stage_iters[cur_stage+1] - self.noise_stage_iters[cur_stage])
        else:
            local_step = 1
        step_ratio = min(1, local_step)

        if self.random_noise_step:
            step_ratio *= 0.5       # assume averaging here

        elif self.annealing_noise_decay:
            step_ratio = math.sqrt(step_ratio)

        if cur_stage < len(self.noise_stage_iters[1:]):
            if hasattr(self, 'noise_decay'):
                if self.noise_decay[cur_stage]:
                    noise_level = self.noise_minmax[cur_stage]
                    noise_max = (noise_level[1] - noise_level[0])*(1-local_step) + noise_level[0]
                else:
                    noise_level = self.noise_minmax[cur_stage]
                    noise_max= noise_level[1]
            else:
                noise_level = self.noise_minmax[cur_stage]
                noise_max= noise_level[1]
        else:
            noise_level = self.noise_minmax[-1]
            noise_max= noise_level[1]

        return avg_step, noise_max

    def get_loss(
                self, 
                gs_rendered_img, pnum, vers, hors, radii, 
                cond_image=[], 
                additional_prompt=None,
                iteration=-1, 
                mask=None, 
                num_inference_steps=10, 
                # ddim_fixed_max_time_step: bool=True, 
                save_intermediate=True, 
                img_description="", 
                minimum_mask_thrs=0.02,
                cfg_rescale_weight=0.8,
                time_step=None,
                ):
        """
        Given rendered image, get guidance loss with ControlNet
        """
        loss_dict = dict()
        
        # ---------------------------------- Noise scheduling Part ----------------------------------
        # get current stage
        if time_step is None:
            cur_stage = self.get_cur_stage(self.step[pnum])
            if cur_stage < len(self.noise_stage_iters[1:]):
                local_step = self.step[pnum] - self.noise_stage_iters[cur_stage]
                local_step = local_step / (self.noise_stage_iters[cur_stage+1] - self.noise_stage_iters[cur_stage])
            else:
                local_step = 1
            step_ratio = min(1, local_step)

            # change step_ratios
            if self.random_noise_step:
                step_ratio = None

            elif self.annealing_noise_decay:
                step_ratio = math.sqrt(step_ratio)

            # set noise levels (it's pnum dependent updates. so we need to change it ALWAYS)
            if cur_stage < len(self.noise_stage_iters[1:]):
                if hasattr(self, 'noise_decay'):
                    if self.noise_decay[cur_stage]:
                        noise_level = self.noise_minmax[cur_stage]
                        noise_max = (noise_level[1] - noise_level[0])*(1-local_step) + noise_level[0]
                        self.guidance_controlnet.set_noise_level([noise_level[0], noise_max])
                        self.noise_max[pnum] = noise_max
                        self.noise_min[pnum] = noise_level[0]
                else:
                    noise_level = self.noise_minmax[cur_stage]
                    self.set_noise_level(noise_level)
                    self.noise_max[pnum] = noise_level[1]
                    self.noise_min[pnum] = noise_level[0]
            else:
                noise_level = self.noise_minmax[-1]
                self.set_noise_level(noise_level)
        else:
            step_ratio = time_step
        
        # Additionally add random mode
        if step_ratio == 1:
            step_ratio = None
        
        # ---------------------------------- cfg scheduling Part ----------------------------------
        # get current stage
        if self.use_cfg_sched:
            cfg_stage = self.get_cur_stage(self.step[pnum], is_noise=False, is_cfg=True)
            
            if cfg_stage < len(self.cfg_stage_iters[1:]):
                local_step = self.step[pnum] - self.cfg_stage_iters[cfg_stage]
                local_step = local_step / (self.cfg_stage_iters[cfg_stage+1] - self.cfg_stage_iters[cfg_stage])
            else:
                local_step = 1
            cfg_step_ratio = min(1, local_step)

            # set noise levels (it's person number dependent updates. so we need to change it ALWAYS)
            if cfg_stage < len(self.cfg_stage_iters[1:]):
                if hasattr(self, 'noise_decay'):
                    if self.cfg_decay[cfg_stage]:
                        cfg_level = self.cfg_minmax[cfg_stage]
                        cfg_scale = (cfg_level[1] - cfg_level[0])*(1-local_step) + cfg_level[0]
                    else:
                        cfg_scale = self.cfg_minmax[cfg_stage][-1]
                else:
                    cfg_scale = self.cfg_minmax[cfg_stage][-1]
            else:
                cfg_scale = self.cfg_minmax[-1][-1]
            
            print(f"[INFO] using cfg scheduling {cfg_scale}")
            self.guidance_scale = cfg_scale

        # ---------------------------------- Calculate loss ----------------------------------
        loss = 0

        loss_imgs = []
        intermediate_masks = None
        
        if self.enable_controlnet:
            new_cond_image = cond_image # image w/ openpose
            guid_loss, t = self.guidance_controlnet.train_step(
                                                                gs_rendered_img, 
                                                                step_ratio=step_ratio, 
                                                                guidance_scale=self.guidance_scale, 
                                                                controlnet_weight=self.controlnet_weight,
                                                                hold_last_img=save_intermediate, 
                                                                cond_image=new_cond_image, 
                                                                additional_prompt=additional_prompt,
                                                                mask=mask,
                                                                pnum=pnum, 
                                                                cfg_rescale_weight=cfg_rescale_weight,
                                                                num_inference_steps=num_inference_steps,
                                                                minimum_mask_thrs=minimum_mask_thrs)
            loss = loss + self.opt.lambda_controlnet * guid_loss

            if save_intermediate:
                loss_imgs.append(self.guidance_controlnet.last_loss_imgs)
                if hasattr(self.guidance_controlnet, "last_inpaint_masks"):
                    intermediate_masks = self.guidance_controlnet.last_inpaint_masks
        
        # ---------------------------------- Log loss images ----------------------------------
        if True:
            # store input images
            input_img = gs_rendered_img.detach().cpu().squeeze().permute(1,2,0).numpy()
            input_img = (input_img * 255).astype(np.uint8)      # hold in RGB format
            self.input_img_dict[pnum].append(input_img)

            if len(self.input_img_dict[pnum]) > 10:
                self.input_img_dict[pnum].pop(0)

            all_imgs = []
            if len(loss_imgs) > 0:
                for loss_img in loss_imgs:
                    # 1. append loss image
                    imgs = [_img for _img in loss_img]
                    imgs = torch.cat(imgs, dim=2)
                    imgs = imgs.squeeze()
                    imgs = imgs.detach().cpu().permute(1,2,0).numpy()
                    imgs = (imgs * 255).astype(np.uint8)
                    imgs = imgs[..., [2,1,0]]    # convert to BGR
                    imgs = img_add_text(imgs.copy(), f"{pnum}_{iteration}_pstep{self.step[pnum]:05}_loss_{loss.detach().item():.6f}_t_{t.item():.3f}")    
                    
                    _img_shape_y = imgs.shape[0]
                    _img_shape_x = imgs.shape[1]

                    # (optional) append visibility mask
                    if mask is not None:
                        mask = mask.squeeze().detach().cpu().unsqueeze(-1).repeat(1,1,3)
                        mask = (mask.numpy() * 255).astype(np.uint8)
                        mask = cv2.resize(mask, (_img_shape_x, _img_shape_y))
                        mask = img_add_text(mask.copy(), f"visiblity mask")    
                        imgs = np.concatenate([imgs, mask], axis=1)
                        
                        _mask = cv2.resize(intermediate_masks, (_img_shape_x, _img_shape_y))
                        _mask = img_add_text(_mask.copy(), f"mask in DGM")    
                        imgs = np.concatenate([imgs, _mask], axis=1)

                    # 2. append cond_image (openpose)
                    if len(cond_image) > 0 and isinstance(cond_image[0], PIL.Image.Image):
                        cond_image = np.array(cond_image[0])
                        cond_image = cv2.resize(cond_image, (_img_shape_x, _img_shape_y))
                        cond_image = cond_image[..., [2,1,0]]
                        if len(img_description) > 0:
                            cond_image = img_add_text(cond_image.copy(), img_description)
                        imgs = np.concatenate([imgs, cond_image], axis=1)

                    # 3. append rendered input_img 
                    input_img = self.input_img_dict[pnum][-1]
                    input_img = cv2.resize(input_img, (_img_shape_x, _img_shape_y))
                    input_img = img_add_text(input_img.copy(), self.guidance_controlnet.prompts[0])
                    input_img = input_img[..., [2,1,0]]
                    imgs = np.concatenate([imgs, input_img], axis=1)

                    all_imgs.append(imgs)

                all_imgs = np.concatenate(all_imgs, axis=0) # stitch all loss images into one image: {loss_img, (mask_img), cond_img, rendered_img}

                self.diffusion_img_dict[pnum].append(all_imgs)

                # save every 100 iterations
                if iteration % 100 == 0:
                    if len(self.diffusion_img_dict[pnum]) >= 1:
                        cv2.imwrite(str(self.diffusion_img_log_dir[pnum] / f"{iteration:05}.jpg"), self.diffusion_img_dict[pnum][-1]) # save recent image
                        if intermediate_masks is not None:
                            cv2.imwrite(str(self.diffusion_img_log_dir[pnum] / f"mask_{iteration:05}.jpg"), intermediate_masks)
                        self.diffusion_img_dict[pnum] = [] # empty list

        # log recent loss
        self.dg_loss_log_stack[pnum].append(loss.detach().cpu().numpy())
        loss = self.dg_lambda[pnum] * loss

        return loss, t, loss_dict