import torch
import numpy as np
from showmak3r.pipeline.smpl_deform.smpl.body_models import SMPL

class SMPLWrapper(torch.nn.Module):
    def __init__(self, smpl_model_path, gender='neutral', scale=1, betas=None, 
        v_template=None, use_hands=False, use_feet_keypoints=False, device="cuda:0"):
        super().__init__()
        self.scale = scale
        self.smpl = SMPL(model_path=smpl_model_path,
                         gender=gender,
                         batch_size=1,
                         use_hands=use_hands,
                         use_feet_keypoints=use_feet_keypoints,
                         dtype=torch.float32).to(device)
        self.bone_parents = self.smpl.bone_parents.astype(int)
        self.bone_parents[0] = -1
        self.bone_ids = []
        for i in range(24): 
            self.bone_ids.append([self.bone_parents[i], i])

        if v_template is not None:
            self.v_template = torch.tensor(v_template).float().to(device)
        else:
            self.v_template = None

        if betas is not None:
            if isinstance(betas, np.ndarray):
                self.betas = torch.tensor(betas).float().to(device)
            else:
                self.betas = betas.float().to(device)
        else:
            self.betas = None

        # define the canonical pose
        param_canonical = torch.zeros((1, 86),dtype=torch.float32).to(device)
        param_canonical[0, 0] = self.scale
        param_canonical[0, 9] = np.pi / 6
        param_canonical[0, 12] = -np.pi / 6
        if self.betas is not None and self.v_template is None:
            param_canonical[0,-10:] = self.betas
        self.param_canonical = param_canonical

        output = self.forward(param_canonical, absolute=True)
        self.verts_c = output['smpl_verts']
        self.weights_c = output['smpl_weights']
        self.joints_c = output['smpl_jnts']
        self.tfs_c_inv = output['smpl_tfs'].squeeze(0).inverse()


    def forward(self, smpl_params, absolute=False):
        """return SMPL output from params

        Args:
            smpl_params : smpl parameters. shape: [B, 86]. [0-scale,1:4-trans, 4:76-thetas,76:86-betas]
            absolute (bool): if true return smpl_tfs wrt thetas=0. else wrt thetas=thetas_canonical. 

        Returns:
            smpl_verts: vertices. shape: [B, 6893. 3]
            smpl_tfs: bone transformations. shape: [B, 24, 4, 4]
            smpl_jnts: joint positions. shape: [B, 25, 3]
        """

        output = {}

        scale, transl, thetas, betas = torch.split(smpl_params, [1, 3, 72, 10], dim=1)

        smpl_output = self.smpl.forward(betas=betas,
                                        transl=torch.zeros_like(transl),
                                        body_pose=thetas[:, 3:],
                                        global_orient=thetas[:, :3],
                                        return_verts=True,
                                        return_full_pose=True,
                                        v_template=None)

        verts = smpl_output.vertices.clone()
        output['smpl_verts'] = verts * scale.unsqueeze(1) + transl.unsqueeze(1)

        faces = torch.tensor(smpl_output.faces.copy()).to(verts.device)
        output['smpl_faces'] = faces
        
        joints = smpl_output.joints.clone()
        output['smpl_jnts'] = joints * scale.unsqueeze(1) + transl.unsqueeze(1)

        tf_mats = smpl_output.T.clone()
        tf_mats[:, :, :3, :] *= scale.unsqueeze(1).unsqueeze(1)
        tf_mats[:, :, :3, 3] += transl.unsqueeze(1)

        if not absolute:
            tf_mats = torch.einsum('bnij,njk->bnik', tf_mats, self.tfs_c_inv)
        
        output['smpl_tfs'] = tf_mats

        output['smpl_weights'] = smpl_output.weights
        
        return output