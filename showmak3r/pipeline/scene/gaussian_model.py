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
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
import open3d as o3d

from showmak3r.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from showmak3r.utils.loss_utils import l2_loss
from showmak3r.utils.sh_utils import RGB2SH
from showmak3r.utils.graphics_utils import BasicPointCloud
from showmak3r.utils.general_utils import strip_symmetric, build_scaling_rotation, rotation_to_quaternion, rotation_matrix_from_vectors, batched_rotation_matrix_from_vectors, batched_rotation_to_quaternion
from showmak3r.utils.sh_utils import SH2RGB, RGB2SH

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
    
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.stacked_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        

        if getattr(training_args, 'do_smpl_mod', False):
            self._additional_setup(training_args)
    
    def _additional_setup(self, training_args):
        '''
        SMPL related setup
        '''
        self.is_smpl_gs = True

        self.fix_init_smpls_verts = training_args.fix_init_smpls_verts
        self.init_xyz_location = self._xyz.data.clone().detach()
        self.init_scales = self._scaling.data.clone().detach()
        self.remaining_init_gs = len(self._opacity)                     # 

        # some of basic settings
        self.allow_init_smpl_splitting = training_args.allow_init_smpl_splitting
        self.allow_init_smpl_cloning = training_args.allow_init_smpl_cloning
        self.allow_init_smpl_pruning = training_args.allow_init_smpl_pruning

        # split gaussian if it's TOO sharp
        self.split_sharp_gaussian = training_args.split_sharp_gaussian

        #### Additional setup for SMPL related modification ######
        self.clip_init_smpl_opacity = training_args.clip_init_smpl_opacity
        self.smpl_opacity_clip_min = training_args.smpl_opacity_clip_min
        self.track_parent_id = training_args.track_gs_parent_id
        self.init_parent_stack()
        
        # Visibility stack
        # self.init_visibility_stack = [[] for _ in range(len(self._xyz))]

    def __init__(self, sh_degree : int, disc_gaussian: bool=False):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.stacked_denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.disc_gaussian = disc_gaussian
        self.setup_functions()

    # ============================== Property functions ==============================
    
    def disc2raw(self, data):        
        if False:
            data = data * torch.tensor([[1., 1., 0.05]]).to(data.device).float()
        else:
            data = torch.cat([data, data.mean(-1).unsqueeze(-1)*0.05], dim=-1)
        return data
    
    def raw2disc(self, data):   
        if False:
            data = data / torch.tensor([[1., 1., 0.05]]).to(data.device).float()
            data = data[...,:2].mean(-1)[..., None]     # do not consider small z-dir issues
        else:
            data = data[..., :2]
        return data

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self.disc_gaussian:
            scaling = self.disc2raw(self._scaling) 
        else:
            scaling = self._scaling

        return self.scaling_activation(scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_n_points(self):
        return len(self._xyz)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def get_pcd(self):
        '''
        get simple point cloud data of gaussian centers
        '''
        xyz = self._xyz
        xyz_np = xyz.cpu().numpy()
        rgb_dc = SH2RGB(self._features_dc)
        rgbs = torch.clamp_min(rgb_dc, 0.0)
        rgbs = torch.clamp_max(rgbs, 1.0)
        rgbs = rgbs[:, 0, :]
        rgbs_np = rgbs.cpu().numpy()
        return xyz_np, rgbs_np
    
    def save_pcd(self, path):
        '''
        save simple point cloud data of gaussian centers
        '''
        xyz, rgbs = self.get_pcd()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgbs)
        o3d.io.write_point_cloud(path, pcd)

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, init_w_normal: bool=False, init_opacity: float=0.1):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print(f"Number of points at initialization : {fused_point_cloud.shape[0]}")

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)

        if self.disc_gaussian:
            scales = self.raw2disc(torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3))
        else:
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        if init_w_normal:
            # use negative z-direction as unit vector
            unit_vector = torch.tensor([0., 0., -1.,]).float().cuda()       # (3)
            normals = torch.tensor(np.asarray(pcd.normals)).float().cuda()  # (N, 3)
            
            batched_rotations = batched_rotation_matrix_from_vectors(unit_vector, normals)
            rots = batched_rotation_to_quaternion(batched_rotations)

            self.init_normal = normals
            
        else:
            rots[:, 0] = 1

        opacities = inverse_sigmoid(init_opacity * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def init_parent_stack(self):
        # Parents_list 
        self.tracked_parents = torch.tensor(list(range(len(self._xyz))), dtype=torch.int32, device="cuda")     # initialize with initial points
    
    def reset_visibility(self):
        # reset trackings
        self.init_parent_stack()
        # get visibility
        self.init_visibility_stack = [[] for _ in range(len(self.tracked_parents))]


    def update_visibility(self, visible_mask, canon_dir_pps):
        assert self.track_parent_id, "It didn't tracked parents IDs!"

        parent_ids = self.tracked_parents[visible_mask]
        # n_parent_ids = len(torch.unique(parent_ids))

        for parent_id, canon_dir_pp in zip(parent_ids, canon_dir_pps):
            parent_id = int(parent_id.item())
            self.init_visibility_stack[parent_id].append(canon_dir_pp)

        # print(f"[INFO] added {len(canon_dir_pps)} view direction into {n_parent_ids}/6890 points")
    def update_visibility_postfix(self, viz_mask_thrs=0.5, is_view_dir_cond=False):
        self.thrs = [[] for _ in range(len(self.init_visibility_stack))]
        for i, viewed_dir in enumerate(self.init_visibility_stack):
            if len(viewed_dir) > 0:
                views = torch.stack(viewed_dir, dim=0)
                mean_view = views.mean(0)
                mean_view = mean_view / ((mean_view ** 2).sum()).sqrt()
                max_var = ((views * mean_view[None]).sum(-1)).min()
                # print(f"{len(views)}, {max_var.item()}")
                 
                self.init_visibility_stack[i] = mean_view.float().to(self._opacity.device)
                if is_view_dir_cond:
                    self.thrs[i] = torch.cos(torch.arccos(max_var) + torch.pi/6).item()
                else:
                    self.thrs[i] = 1
                # print(self.thrs[i])
            else:
                if is_view_dir_cond:
                    self.init_visibility_stack[i] = torch.zeros(3).float().to(self._opacity.device)
                    self.thrs[i] = 1
                else:
                    self.init_visibility_stack[i] = torch.zeros(1).float().to(self._opacity.device)
                    self.thrs[i] = 1
        
        self.init_visibility_stack = torch.stack(self.init_visibility_stack)
        self.thrs = torch.tensor(self.thrs).to(self._opacity.device)

        self.viz_mask_thrs = viz_mask_thrs
    
    def get_baseline_colors(self):
        n_init_verts = self.init_xyz_location.shape[0]
        self.smpl_emfs = [[] for _ in range(n_init_verts)]

        # similar from Angular EMF, we calculate such values
        # the value range is from (0~pi) and unseen is -1. 
        # Only for SMPL vertices.
        for i in range(n_init_verts):
            viewed_dir = self.init_visibility_stack[i]
            if len(viewed_dir) > 1:
                views = torch.stack(viewed_dir, dim=0)      # (N, 3)
                # get inner products 
                inner_product = views @ views.T             # (N, N)
                min_inner_product = inner_product.reshape(-1).min()     # single value
                self.smpl_emfs[i] = torch.acos(min_inner_product)

                print(f"N view: {len(viewed_dir)}, max: {inner_product.max()}, min:{inner_product.min()}")


            elif len(viewed_dir) == 1:
                self.smpl_emfs[i] = -torch.zeros(1).float().to(self._opacity.device)
            else:
                self.smpl_emfs[i] = -torch.ones(1).float().to(self._opacity.device)
        self.smpl_emfs = torch.tensor(self.smpl_emfs).to(self._opacity.device)

    def get_viz_mask(self, canon_dir_pps, non_directional_vizmask: bool=False, sine_smoothing: bool=False, is_view_dir_cond=False):
        if False:
            canon_dir_pps = canon_dir_pps.cpu()
            viz_mask = []
            for parent_id, canon_dir_pp in zip(self.tracked_parents, canon_dir_pps):
                viz_stack = self.init_visibility_stack[parent_id]
                if viz_stack is None:
                    nearest_angle = 0
                elif non_directional_vizmask:
                    nearest_angle = 1
                else:
                    nearest_angle = (viz_stack @ canon_dir_pp).min().item()
                    nearest_angle = 1 if nearest_angle > 1 else nearest_angle
                    nearest_angle = 0 if nearest_angle < 0 else nearest_angle
                    
                    # Smoothing.
                    nearest_angle = torch.tensor([nearest_angle], dtype=torch.float32)
                    if sine_smoothing:
                        nearest_angle = torch.sin(nearest_angle * torch.pi / 2)
                
                viz_mask.append(nearest_angle)
            
            viz_mask = torch.tensor(viz_mask, dtype=torch.float32).cuda()
            inviz_mask = 1 - viz_mask
        elif is_view_dir_cond:
            nearest_angle = torch.einsum('bi,bi->b', self.init_visibility_stack, canon_dir_pps)
            inviz_mask = nearest_angle < (self.thrs)        # cos(30)
            inviz_mask = inviz_mask.float()
        else:
            inviz_mask =  (self.init_visibility_stack.squeeze() == 0)
            
        return inviz_mask


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        """
        Helper function, to save points in ply
        """
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.get_scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        if hasattr(self, 'tracked_parents'):
            l.append('parent')
        return l

    def save_ply(self, path, lbs_weights=None):
        """
        lbs_weights : [N, J]
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self.get_scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        if hasattr(self, 'tracked_parents'):
            parents = self.tracked_parents.detach().cpu().numpy()
            parents = parents[..., None]

        attr_lists = self.construct_list_of_attributes()
        if lbs_weights is not None:
            for i in range(lbs_weights.shape[-1]):
                attr_lists.append('lbs_w_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in attr_lists]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        _attributes = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]
        if hasattr(self, 'tracked_parents'):
            _attributes.append(parents)
        if lbs_weights is not None:
            if isinstance(lbs_weights, torch.Tensor):
                lbs_weights = lbs_weights.detach().cpu().numpy()        # [N, J]
            _attributes.append(lbs_weights)
        
        attributes = np.concatenate(_attributes, axis=1)
            
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, only_visible=False):
        if only_visible:
            if False:
                target = self.thrs < torch.cos(torch.ones_like(self.thrs)* torch.pi/5)
                neg_target = self.thrs >= 1/(torch.ones(1).float().to(self._opacity.device)*2).sqrt()
                print(f"n_reseting_target: {target.sum()} / {len(target)}")
            else:
                neg_target = self.init_visibility_stack.squeeze() > self.viz_mask_thrs

                
            opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
            opacities_new[neg_target] = self._opacity[neg_target]
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]
        else:
            opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, do_inverse_activation=True, is_insert=False):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        if not is_insert:
            assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        else:
            self.max_sh_degree = 2
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            if do_inverse_activation:
                scales[:, idx] = self.scaling_inverse_activation(torch.from_numpy(np.asarray(plydata.elements[0][attr_name]))).numpy()
            else:
                scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if self.disc_gaussian:
            scales = torch.from_numpy(scales)
            scales = self.raw2disc(scales).numpy()

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        

        if "parent" in plydata.elements[0]:
            parents = np.asarray(plydata.elements[0]["parent"])
            self.tracked_parents = torch.tensor(parents, dtype=torch.int32, device="cuda") 

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def prune_gradients(self, mask, only_rgb=True):
        for group in self.optimizer.param_groups:
            if only_rgb:
                if group["name"] not in ["f_dc", "f_rest"]:
                    continue
            group['params'][0].grad[mask] = 0
        

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        
        if getattr(self, 'fix_init_smpls_verts', False) or (not getattr(self, 'allow_init_smpl_pruning', False)):
            if getattr(self, 'is_smpl_gs', False):
                # Instead reduce the scale of corresponding points
                with torch.no_grad():
                    self._scaling.data[:self.remaining_init_gs][mask[:self.remaining_init_gs]] = self.init_scales[mask[:self.remaining_init_gs]]

                    opacities_new = inverse_sigmoid(torch.ones_like(self._opacity.data[:self.remaining_init_gs][mask[:self.remaining_init_gs]])*0.01)
                    self._opacity.data[:self.remaining_init_gs][mask[:self.remaining_init_gs]] = opacities_new

                mask[:self.remaining_init_gs] = False   # change to save initial mask
        
        
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        if getattr(self, 'track_parent_id', False) and hasattr(self, 'tracked_parents'):
            self.tracked_parents = self.tracked_parents[valid_points_mask]

        if hasattr(self, 'stacked_denom'):
            self.stacked_denom = self.stacked_denom[valid_points_mask]

        
        # Check whether initial point is included in pruning points
        if hasattr(self, 'remaining_init_gs') and getattr(self, 'is_smpl_gs', False):
            n_pruned = (mask[:self.remaining_init_gs]).sum()

            # save the number of pruned points
            self.init_xyz_location = self.init_xyz_location[valid_points_mask[:self.remaining_init_gs]]
            self.init_scales = self.init_scales[valid_points_mask[:self.remaining_init_gs]]
            self.remaining_init_gs -= n_pruned


        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        if hasattr(self, 'init_visibility_stack'):
            self.init_visibility_stack = self.init_visibility_stack[valid_points_mask]
        if hasattr(self, 'thrs'):
            self.thrs = self.thrs[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # ============================== Densification related functions ==============================
    
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_parents=None, new_stacked_denom=None, new_thrs=None, new_visibility=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if getattr(self, 'track_parent_id', False) and hasattr(self, 'tracked_parents') and not (new_parents is None):
            self.tracked_parents = torch.cat([self.tracked_parents, new_parents], dim=0)

        if hasattr(self, 'stacked_denom') and not (new_stacked_denom is None):
            self.stacked_denom = torch.cat([self.stacked_denom, new_stacked_denom], dim=0)
            
        if hasattr(self, 'thrs') and not (new_thrs is None):
            self.thrs = torch.cat([self.thrs, new_thrs], dim=0)
            
        if hasattr(self, 'init_visibility_stack') and not (new_visibility is None):
            self.init_visibility_stack = torch.cat([self.init_visibility_stack, new_visibility], dim=0)


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, d_opacity=0.):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if not getattr(self, 'allow_init_smpl_splitting', False) and getattr(self, 'is_smpl_gs', False):
            selected_pts_mask[:self.remaining_init_gs] = False   # change to save initial mask

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_stacked_denom = self.stacked_denom[selected_pts_mask].repeat(N,1)
        if isinstance(d_opacity, torch.Tensor):
            new_d_opacity = d_opacity[selected_pts_mask].repeat(N,1)
            new_d_opacity = torch.cat([d_opacity, new_d_opacity], dim=0)
        else:
            new_d_opacity = None

        if self.disc_gaussian:
            new_scaling = self.raw2disc(new_scaling)
        
        new_parents = None
        if getattr(self, 'track_parent_id', False) and hasattr(self, 'tracked_parents'):
            new_parents = self.tracked_parents[selected_pts_mask].repeat(N)
            
        new_thrs = None
        if hasattr(self, 'thrs'):
            new_thrs = self.thrs[selected_pts_mask].repeat(N)
        new_visibility = None
        if hasattr(self, 'init_visibility_stack'):
            new_visibility = self.init_visibility_stack[selected_pts_mask].repeat(N, 1)  
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_parents, new_stacked_denom, new_thrs, new_visibility)

        return new_d_opacity

        
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

        return new_d_opacity

    def densify_and_clone(self, grads, grad_threshold, scene_extent, d_opacity=0.):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        if not getattr(self, 'allow_init_smpl_cloning', False) and getattr(self, 'is_smpl_gs', False):
            selected_pts_mask[:self.remaining_init_gs] = False   # change to save initial mask
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_stacked_denom = self.stacked_denom[selected_pts_mask]
        if isinstance(d_opacity, torch.Tensor):
            new_d_opacity = d_opacity[selected_pts_mask]
            new_d_opacity = torch.cat([d_opacity, new_d_opacity], dim=0)
        else:
            new_d_opacity = None

        new_parents = None
        if getattr(self, 'track_parent_id', False) and hasattr(self, 'tracked_parents'):
            new_parents = self.tracked_parents[selected_pts_mask]
            
        new_thrs = None
        if hasattr(self, 'thrs'):
            new_thrs = self.thrs[selected_pts_mask]
        new_visibility = None
        if hasattr(self, 'init_visibility_stack'):
            new_visibility = self.init_visibility_stack[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_parents, new_stacked_denom, new_thrs, new_visibility)

        return new_d_opacity
    
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, d_opacity=0., delete_unseen=False, delete_large=False, iter=-1):
        """
        Main function, that modifying the topology of gaussians
        """

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        if isinstance(d_opacity, torch.Tensor):
            _d_opacity = self.densify_and_clone(grads, max_grad, extent, d_opacity = d_opacity)
            _d_opacity = self.densify_and_split(grads, max_grad, extent, d_opacity = _d_opacity)
            person_d_opacity = self.get_opacity + _d_opacity
        else:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)
            person_d_opacity = self.get_opacity
            
        prune_mask = (person_d_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        if delete_unseen:
            unseen_part = (self.stacked_denom == 0).squeeze()
            prune_mask = torch.logical_or(prune_mask, unseen_part)

        if delete_large:
            large_mask = self.get_scaling.max(dim=1).values > 0.05 * extent
            prune_mask = torch.logical_or(prune_mask, large_mask)

        self.prune_points(prune_mask)

        if getattr(self, 'split_sharp_gaussian', False) and getattr(self, 'is_smpl_gs', False):
            self.split_sharp_gaussians()
            
        torch.cuda.empty_cache()
        
    def prune_gaussians(self, min_opacity, d_opacity=0., extent=None, max_screen_size=None, min_scale=None):
        if isinstance(d_opacity, torch.Tensor):
            person_d_opacity = self.get_opacity + d_opacity
        else:
            person_d_opacity = self.get_opacity
        
        prune_mask = (person_d_opacity < min_opacity).squeeze()
        
        if extent is not None:
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        
        if max_screen_size is not None:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

        if min_scale is not None:
            small_gaussian_mask = self.max_radii2D < min_scale
            prune_mask = torch.logical_or(prune_mask, small_gaussian_mask)
        
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor=None, update_filter=None, viewspace_point_tensor_grad=None):
        if viewspace_point_tensor is None:
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor_grad[update_filter,:2], dim=-1, keepdim=True)
        else: 
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        self.stacked_denom[update_filter] += 1
        
        
    def add_gaussians(self, adding_gaussians):
        """
        stitch gaussians, for joint rendering
        """
        new_xyz = adding_gaussians._xyz
        new_features_dc = adding_gaussians._features_dc
        new_features_rest = adding_gaussians._features_rest
        new_opacity = adding_gaussians._opacity
        new_scaling = adding_gaussians._scaling
        new_rotation = adding_gaussians._rotation

        if self.disc_gaussian:
            new_scaling = self.raw2disc(new_scaling)


        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    
    # ============================== SMPL related functions ==============================
    def split_sharp_gaussians(self, ratio_thrs=10., N=2):
        """
        ratio_thrs: if largest STD > 2nd largest STD * 5 -> split into N gaussians
        """
        selected_pts_mask = torch.where(torch.max(self.get_scaling, dim=1).values > torch.topk(self.get_scaling, k=2, dim=1).values[:, 1] * ratio_thrs, True, False)
        
        # Just copied split code naively (it's to complex to consider new aprroach)
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)    
    
    def fix_smpl_init_position(self, reset_scale=True):
        """
        Fix SMPL position to initals. 
        """
        
        self._xyz[:self.remaining_init_gs] = self.init_xyz_location

        if reset_scale:
            self._scaling[:self.remaining_init_gs] = self.init_scales


        if self.clip_init_smpl_opacity:
            # print("clipping opacities")
            opacities_new = inverse_sigmoid(torch.max(self.get_opacity, torch.ones_like(self.get_opacity)*self.smpl_opacity_clip_min))

            for group in self.optimizer.param_groups:
                if group["name"] == "opacity":
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    if stored_state is None:
                        print("Not optimized yet in this loop")
                        return
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]
        
        
    def get_smpl_vert_loss(self):
        """
        Calculate L2 distance between fixed init position <-> current positions
        """
        cur_xyz = self._xyz[:self.remaining_init_gs]
        loss = l2_loss(cur_xyz, self.init_xyz_location)
        
        return loss


    def clip_invalid_shs(self):
        """
        To avoid oversaturation, clip too large features.
        (It shouldn't be happen in raw 3D-GS but ours use single GS & large Loss, so )
        """
        feature_dc = self._features_dc.detach()

        rgb_dc = SH2RGB(feature_dc)

        n_invalid_pts = (((rgb_dc < 0.) + (rgb_dc>1.)).sum(-1) > 0).sum().detach().cpu()

        rgbs = torch.clamp_min(rgb_dc, 0.0)
        rgbs = torch.clamp_max(rgbs, 1.0)
        feature_dc = RGB2SH(rgbs)

        # clamp rgbs
        self._features_dc += feature_dc - self._features_dc

        return n_invalid_pts
        

    def prune_infnan_points(self, offset=None, scale=None):
        """
        we also remove points > LBS defined bbox

        offset: [1, 3]
        scale: [1, 3]
        """
        scaling_mask = (self._scaling.isnan() + self._scaling.isinf()).sum(-1) > 0
        rotating_mask = (self.get_rotation.isnan() + self.get_rotation.isinf()).sum(-1) > 0
        opacity_mask = (self.get_opacity.isnan() + self.get_opacity.isinf()).sum(-1) > 0
        feature_mask = (self.get_features.isnan() + self.get_features.isinf()).sum([1,2]) > 0
        
        nan_mask = (scaling_mask + rotating_mask + opacity_mask + feature_mask) > 0
        nan_mask = nan_mask.squeeze()

        if getattr(self, 'is_smpl_gs', False) and (offset is not None) and (scale is not None):
            xyzs = self.get_xyz
            xyzs = (xyzs + offset) * scale
            xyz_mask = ((xyzs > 1.0) + (xyzs < -1.0)).sum(-1).squeeze() > 0
            nan_mask = (nan_mask + xyz_mask) > 0
    
        n_nan = nan_mask.sum()
        
        if n_nan > 0:
            restore = False
            if getattr(self, 'fix_init_smpls_verts', False) or (not getattr(self, 'allow_init_smpl_pruning', False)):
                _fix_init_smpl_verts = getattr(self, 'fix_init_smpls_verts', False)
                _allow_init_smpl_pruning = getattr(self, 'allow_init_smpl_pruning', False)
                self.fix_init_smpls_verts = False
                self.allow_init_smpl_pruning = True

                restore = True

            print(f"[INFO] Pruning {n_nan} NaN points heres. ({self.get_n_points} -> {self.get_n_points - n_nan})")
            self.prune_points(nan_mask)

            if restore:
                self.fix_init_smpls_verts = _fix_init_smpl_verts
                self.allow_init_smpl_pruning = _allow_init_smpl_pruning
            
        torch.cuda.empty_cache()
            
        return n_nan
        
    def turn_off(self):
        prune_mask = torch.zeros_like(self.get_opacity).to(self.get_opacity.device).squeeze()
        prune_mask[:1] = torch.ones_like(prune_mask[:1]).to(prune_mask[:1].device)
        prune_mask = prune_mask > 0.5

        # turn off single remaining PCs
        # opacities_new = self._opacity.clone().detach()
        opacities_new = torch.ones_like(self._opacity) * (-10000)
        # optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = opacities_new
        self.prune_points(prune_mask)


        print(f"\n\n\n[INFO] Turn off the Gaussians (turn off) \n\n\n")
        
        