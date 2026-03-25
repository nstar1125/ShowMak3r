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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from chamfer_distance import ChamferDistance as chamfer_dist
import time
from lpips import LPIPS

# define LPIPS loss here
if torch.cuda.is_available():
    LPIPS_LOSS = LPIPS(net='vgg').cuda()
else:
    LPIPS_LOSS = LPIPS(net='vgg')


def denisty_reg_loss(alpha):
    OFFSET = 0.313262
    reg_alpha = (-torch.log(torch.exp(-alpha) + torch.exp(alpha - 1))).mean() + OFFSET
    
    return reg_alpha


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


chd = chamfer_dist()

def depth_loss_dpt(pred_depth, gt_depth, weight=None):
    """
    :param pred_depth:  (H, W)
    :param gt_depth:    (H, W)
    :param weight:      (H, W)
    :return:            scalar
    """
    
    t_pred = torch.median(pred_depth)
    s_pred = torch.mean(torch.abs(pred_depth - t_pred))

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))

    pred_depth_n = (pred_depth - t_pred) / s_pred
    gt_depth_n = (gt_depth - t_gt) / s_gt

    if weight is not None:
        loss = F.mse_loss(pred_depth_n, gt_depth_n, reduction='none')
        loss = loss * weight
        loss = loss.sum() / (weight.sum() + 1e-8)
    else:
        loss = F.mse_loss(pred_depth_n, gt_depth_n)
    return loss



def get_cd_loss(gt_pixel_list, rendered_pixel_list):    
    # check https://github.com/otaheri/chamfer_distance
    p1 =gt_pixel_list.to(rendered_pixel_list.device).squeeze().unsqueeze(0)
    p2 =rendered_pixel_list.squeeze().unsqueeze(0)
    
    
    # s = time.time()
    dist1, dist2, _, _ = chd(p1,p2)
    # loss = (torch.mean(dist1)) + (torch.mean(dist2))
    loss = (torch.sum(dist1)) + (torch.sum(dist2))

    # torch.cuda.synchronize()
    # print(f"Chamfer distance loss Time: {time.time() - s} seconds")
    
    return loss


def get_lpips_loss(gt_img, target_img):
    # assume value input 0~1
    gt_img = gt_img * 2.0 - 1.0
    target_img = target_img * 2.0 - 1.0

    if len(gt_img.shape) == 3:
        gt_img = gt_img[None]
    
    if len(target_img.shape) == 3:
        target_img = target_img[None]

    percep_loss = LPIPS_LOSS(
        gt_img, 
        target_img
    )

    return percep_loss

def log_l1(pred, gt):
    return torch.log(1 + torch.abs(pred - gt)).mean()

def tv_loss(pred):
    h_diff = pred[..., :, :-1, :] - pred[..., :, 1:, :]
    w_diff = pred[..., :-1, :, :] - pred[..., 1:, :, :]
    return torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(w_diff))

def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared =  x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

def huber_loss(pred, gt, delta=1.0):
    if torch.abs(pred - gt) <= delta:
        return 0.5 * (pred - gt) ** 2
    else:
        return delta * (torch.abs(pred - gt) - 0.5 * delta)

def denisty_reg_loss(alpha):
    OFFSET = 0.313262
    reg_alpha = (-torch.log(torch.exp(-alpha) + torch.exp(alpha - 1))).mean() + OFFSET
    
    return reg_alpha