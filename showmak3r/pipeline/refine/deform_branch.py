import torch
import torch.nn as nn
import torch.nn.functional as F
from showmak3r.pipeline.refine.deform_net import DeformNetwork
import os
from showmak3r.utils.system_utils import searchForMaxIteration
from showmak3r.utils.general_utils import get_expon_lr_func, get_linear_noise_func

def get_residuals(people_infos, deform, fid, total_frame):
    time_interval = 1 / total_frame
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

    means3D_canonical_people = []
    for pi in people_infos:
        person_xyz = pi.gaussians.get_xyz
        means3D_canonical_people.append(person_xyz)
    if person_xyz is not None:
        person_batch_gaussians = torch.cat(means3D_canonical_people, dim=0).contiguous() # concat all gaussians
        N = person_batch_gaussians.shape[0] # number of gaussians
        
        time_step = int(fid.split('_')[-1])
        time_input = torch.tensor(time_step).unsqueeze(0).expand(N, -1).cuda()
        # ast_noise = torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
        ast_noise = 0
        d_color, d_opacity = deform.step(person_batch_gaussians.detach(), time_input + ast_noise)   
    else:
        d_color, d_opacity = 0.0, 0.0
    return d_color, d_opacity

class DeformModel:
    def __init__(self):
        self.deform = DeformNetwork().cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, xyz, time_emb):
        return self.deform(xyz, time_emb)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, output_path):
        torch.save(self.deform.state_dict(), os.path.join(output_path, 'deform.pth'))

    def load_weights(self, output_path):
        weights_path = os.path.join(output_path, "deform.pth")
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
