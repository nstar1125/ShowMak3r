
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.cpp_extension import load


from showmak3r.pipeline.smpl_deform.smpl_wrapper import SMPLWrapper
from showmak3r.pipeline.smpl_deform.helpers import hierarchical_softmax, skinning, bmv, create_voxel_grid, query_weights_smpl
from showmak3r.pipeline.smpl_deform.trans_grid import ImplicitNetwork

from showmak3r.utils.system_utils import searchForMaxIteration
from showmak3r.utils.general_utils import strip_symmetric, unstrip_symmetric


class SMPLDeformer(torch.nn.Module):
    """
    Tensor shape abbreviation:
        B: batch size
        N: number of points
        J: number of bones
        I: number of init
        D: space dimension
    """

    def __init__(self, res: int=64, z_ratio: float=1, global_scale: float=1., smpl_scale: float=1., gender='neutral', beta=None):
        super().__init__()

        self.skinning_mode = 'preset'
        self.res = res
        self.z_ratio = z_ratio
        self.global_scale = global_scale
        self.soft_blend = 20
        self.align_corners = True

        self.init_bones = [0, 1, 2, 4, 5, 16, 17, 18, 19] 
        self.init_bones_cuda = torch.tensor(self.init_bones).cuda().int()

        # convert to voxel grid
        self.smpl_server = SMPLWrapper("submodules/ml-comotion/src/comotion_demo/data/smpl/SMPL_NEUTRAL.pkl", gender=gender, betas=beta, scale=smpl_scale)
        
        smpl_verts = self.smpl_server.verts_c
        device = self.smpl_server.verts_c.device
        self.device = device

        d, h, w = self.res//self.z_ratio, self.res, self.res
        grid = create_voxel_grid(d, h, w, device=device)
        
        gt_bbox = torch.cat([smpl_verts.min(dim=1).values, 
                             smpl_verts.max(dim=1).values], dim=0)
        self.offset = -(gt_bbox[0] + gt_bbox[1])[None,None,:] / 2

        self.scale = torch.zeros_like(self.offset)
        self.scale[...] = 1./((gt_bbox[1] - gt_bbox[0]).max()/2 * self.global_scale)
        self.scale[:,:,-1] = self.scale[:,:,-1] * self.z_ratio

        self.grid_denorm = grid/self.scale - self.offset

        if self.skinning_mode == 'preset':
            self.lbs_voxel_final = query_weights_smpl(self.grid_denorm, smpl_verts, self.smpl_server.weights_c)
            self.lbs_voxel_final = self.lbs_voxel_final.permute(0,2,1).reshape(1,-1,d,h,w)

        elif self.skinning_mode == 'voxel':
            lbs_voxel = 0.001 * torch.ones((1, 24, d, h, w), dtype=self.grid_denorm.dtype, device=self.grid_denorm.device)
            self.register_parameter('lbs_voxel', torch.nn.Parameter(lbs_voxel,requires_grad=True))

        else:
            raise NotImplementedError('Unsupported Deformer.')
        
        self.apply_trans_grid = False
        self.trans_grid = None
        self.last_trans = []
        


    def activate_trans_grid(self, n_frames=-1, trans_grid_lr=1e-4):
        self.n_frames = n_frames
        self.apply_trans_grid = True

        if self.n_frames > 0 and self.apply_trans_grid:
            print(f"\n\n[INFO] Here we use warping function for per-frame frame deformation over-fitting\n\n")
            trans_grid = ImplicitNetwork(
                d_in = 3,
                d_out = 3,
                width = 128,
                depth = 4,
                multires = 6,
                pose_cond_layer = [0],
                pose_cond_dim = 69,
                zero_conv = True,
            )
            trans_grid = trans_grid.to(self.device)
            trans_grid.train()
            self.trans_grid_optimizer = torch.optim.Adam(trans_grid.parameters(), lr=trans_grid_lr)
            self.trans_grid = trans_grid
        else:
            self.trans_grid = None
            self.trans_grid_optimizer = None


        return self.trans_grid, self.trans_grid_optimizer


    def dump_trans_grid(self, path: Path):
        if self.trans_grid is not None:
            torch.save(self.trans_grid.state_dict(), path / "deformer_warp_field.pt")
            
        beta = self.smpl_server.betas
        if beta is not None:
            beta = beta.squeeze().clone().detach().cpu().numpy()
            np.save(path / "mean_shape.npy", beta)


    def load_trans_grid(self, path: Path, load_iteration=-1, trans_grid_lr=1e-4):
        if load_iteration:
            if load_iteration == -1:
                loaded_iter = searchForMaxIteration(str(path / "point_cloud"))
            else:
                loaded_iter = load_iteration

            path = path / "point_cloud" / ("iteration_" + str(loaded_iter))
        else:
            path = path

        print(f"[INFO] finding trans field in {str(path)}")
        if (path / "deformer_warp_field.pt").exists():
            trans_grid = ImplicitNetwork(
                d_in = 3,
                d_out = 3,
                width = 128,
                depth = 4,
                multires = 6,
                pose_cond_layer = [0],
                pose_cond_dim = 69,
                zero_conv = True,
            )
            trans_grid_dict = torch.load(path / "deformer_warp_field.pt")
            trans_grid.load_state_dict(trans_grid_dict)
            trans_grid = trans_grid.to(self.device)
            trans_grid.train()
            self.trans_grid_optimizer = torch.optim.Adam(trans_grid.parameters(), lr=trans_grid_lr)
            self.trans_grid = trans_grid
            self.apply_trans_grid = True

            print(f"[INFO] Successfully loaded trans field from {str(path)}")


    def deform_gp(self, pts, cov_3d, smpl_params, cond=None, rotations=None):
        """
        pts: 3D pts centers
        cov_3D: [N, 3, 3]
        """
        # 0. apply trans-grid first
        pts = pts.unsqueeze(0)      # to make batch
        if self.trans_grid is not None and self.apply_trans_grid:
            xc_norm = (pts + self.offset) * self.scale

            if 'thetas' in cond:
                trans_grid_cond = dict(
                    thetas=cond['thetas'].reshape(1, -1)
                )

            elif smpl_params is None:
                trans_grid_cond = dict(
                    thetas=self.smpl_server.param_canonical[:,7:-10]
                )
            else:
                trans_grid_cond = dict(
                    thetas=smpl_params[:,7:-10]
                )
            trans = self.trans_grid(xc_norm, trans_grid_cond)  # [B, N, 3]
            pts = pts + trans
        else:
            trans = 0

        self.last_trans.append(trans)

        # 2. do LBS deformation (but sample before deformation, as deform is for local-transformation)
        if smpl_params is not None:
            smpl_output = self.smpl_server(smpl_params, absolute=False)
            tfs = smpl_output['smpl_tfs']
            xd, w_tf = self.forward_skinning(pts, cond, tfs)

            # rotate the rotation matrix
            R = w_tf[0, :, :3, :3]
            cov_3d = unstrip_symmetric(cov_3d)
            # cov_3d = R @ cov_3d @ R.transpose(1,2)
            
            # Already, R hold scaling factor
            # As covariance doesn't matter the rotation. we shouldn't modify it.
            cov_3d = torch.bmm(torch.bmm(R, cov_3d), R.transpose(1,2)) 
            cov_3d = strip_symmetric(cov_3d)
            
            rotations = R / smpl_params[0,0]
            
            # need to scale covariance matrix with scale of smpl param!
            # scale = smpl_params[0, 0]
            # cov_3d = cov_3d * scale
            
        else:
            xd = pts
            rotations = None
        
        xd = xd
        
        return xd.squeeze(0), cov_3d, rotations
    

    def get_rotations(self, pts, smpl_params, cond=None):
        pts = pts.unsqueeze(0)      # to make batch

        # Convert back it into original space. (as it's angle, we only need to handle Rotations, here)
        if (not self.trans_grid is None) and self.trans_grid.shape[1] > 3:
            raise NotImplementedError("Rotation trans-grid isn't prepared to handle visibility mask")
        

        smpl_output = self.smpl_server(smpl_params, absolute=False)
        tfs = smpl_output['smpl_tfs']
        xd, w_tf = self.forward_skinning(pts, cond, tfs)

        # rotate the rotation matrix
        R = w_tf[0, :, :3, :3] / smpl_params[0,0]

        return R
    

    def forward_skinning(self, xc, cond, tfs, mask=None):
        """Canonical point -> deformed point

        Args:
            xc (tensor): canonoical points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xd (tensor): deformed point. shape: [B, N, D]
        """
        if mask is None:
            w = self.query_weights(xc, cond)
            xd, w_tf = skinning(xc, w, tfs, inverse=False)
        else:
            w = self.query_weights(xc, cond, mask=mask.flatten(0,1))
            xd, w_tf = skinning(xc, w, tfs, inverse=False)

        return xd, w_tf

    def mlp_to_voxel(self):

        d, h, w = self.res//self.z_ratio, self.res, self.res

        lbs_voxel_final = self.lbs_network(self.grid_denorm, {}, None)
        lbs_voxel_final = self.soft_blend * lbs_voxel_final

        if self.softmax_mode == "hierarchical":
            lbs_voxel_final = hierarchical_softmax(lbs_voxel_final)
        else:
            lbs_voxel_final = F.softmax(lbs_voxel_final, dim=-1)

        self.lbs_voxel_final = lbs_voxel_final.permute(0,2,1).reshape(1,24,d,h,w)

    def voxel_to_voxel(self):

        lbs_voxel_final = self.lbs_voxel*self.soft_blend

        self.lbs_voxel_final = F.softmax(lbs_voxel_final, dim=1)

    def query_weights(self, xc, cond=None, mask=None, mode='bilinear'):

        if not hasattr(self,"lbs_voxel_final"):
            if self.skinning_mode == 'mlp':
                self.mlp_to_voxel()
            elif self.skinning_mode == 'voxel':
                self.voxel_to_voxel()

        xc_norm = (xc + self.offset) * self.scale
        
        w = F.grid_sample(self.lbs_voxel_final.expand(xc.shape[0],-1,-1,-1,-1), xc_norm.unsqueeze(2).unsqueeze(2), align_corners=self.align_corners, mode=mode, padding_mode='zeros')
        
        w = w.squeeze(-1).squeeze(-1).permute(0,2,1)
        
        return w
