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
import sys
from datetime import datetime

import numpy as np
import random
import cv2

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    # resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(pil_image)) / 255.0
    if len(resized_image.shape) == 3:
        _return = resized_image.permute(2, 0, 1)
    else:
        _return = resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    _return = torch.nn.functional.interpolate(_return[None], (resolution[1], resolution[0]), mode="bilinear", align_corners=False)[0]
    return _return

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def get_linear_noise_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = lr_init * (1 - t) + lr_final * t
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] += L[:, 0, 0]
    uncertainty[:, 1] += L[:, 0, 1]
    uncertainty[:, 2] += L[:, 0, 2]
    uncertainty[:, 3] += L[:, 1, 1]
    uncertainty[:, 4] += L[:, 1, 2]
    uncertainty[:, 5] += L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def unstrip_symmetric(sym):
    L = torch.zeros((sym.shape[0], 3, 3), dtype=torch.float, device="cuda")

    L[:, 0, 0] += sym[:, 0]
    L[:, 0, 1] += sym[:, 1]
    L[:, 1, 0] += sym[:, 1]
    L[:, 0, 2] += sym[:, 2]
    L[:, 2, 0] += sym[:, 2]
    L[:, 1, 1] += sym[:, 3]
    L[:, 1, 2] += sym[:, 4]
    L[:, 2, 1] += sym[:, 4]
    L[:, 2, 2] += sym[:, 5]

    return L 




# def rotation_matrix_from_vectors(input_vector, desired_vector):
#     input_vector = input_vector / torch.norm(input_vector)
#     desired_vector = desired_vector / torch.norm(desired_vector)
    
#     dot_product = torch.dot(input_vector, desired_vector)
#     cross_product_matrix = torch.tensor([
#         [0, -desired_vector[2], desired_vector[1]],
#         [desired_vector[2], 0, -desired_vector[0]],
#         [-desired_vector[1], desired_vector[0], 0]
#     ], dtype=torch.float32).to(desired_vector.device)
    
#     rotation_matrix = torch.eye(3, dtype=torch.float32).to(desired_vector.device) + cross_product_matrix + \
#                       (1 / (1 + dot_product)) * torch.matmul(cross_product_matrix, cross_product_matrix)
    
#     return rotation_matrix
    
def batched_rotation_matrix_from_vectors(input_vector, desired_vector):
    if len(input_vector.shape) == 1:
        input_vector = input_vector.unsqueeze(0)
    if len(desired_vector.shape) == 1:
        desired_vector = desired_vector.unsqueeze(0)
        
    if desired_vector.shape[0] != input_vector.shape[0]:
        input_vector = input_vector.repeat(desired_vector.shape[0], 1)
    
    
    input_vector = input_vector / torch.norm(input_vector, dim=-1, keepdim=True)                # (B, 3)
    desired_vector = desired_vector / torch.norm(desired_vector, dim=-1, keepdim=True)          # (B, 3)
    
    rotation_axis = torch.cross(input_vector, desired_vector, dim=-1)                           # (B, 3)                           
    rotation_axis = rotation_axis / torch.norm(rotation_axis, dim=-1, keepdim=True)             # (B, 3)
    
    dot_product = torch.einsum('bi,bi->b', input_vector, desired_vector)
    
    angle = torch.acos(dot_product)
    
    c = torch.cos(angle)
    s = torch.sin(angle)
    t = 1 - c
    
    # define rot_matrix
    rot_1 = torch.stack(
        [
            t * rotation_axis[:, 0] * rotation_axis[:, 0] + c, 
            t * rotation_axis[:, 0] * rotation_axis[:, 1] - s * rotation_axis[:, 2], 
            t * rotation_axis[:, 0] * rotation_axis[:, 2] + s * rotation_axis[:, 1]
            ],
        dim=-1
    )
    rot_2 = torch.stack(
        [
            t * rotation_axis[:,0] * rotation_axis[:,1] + s * rotation_axis[:,2], 
            t * rotation_axis[:,1] * rotation_axis[:,1] + c, 
            t * rotation_axis[:,1] * rotation_axis[:,2] - s * rotation_axis[:,0]
            ],
        dim=-1
    )
    rot_3 = torch.stack(
        [
            t * rotation_axis[:,0] * rotation_axis[:,2] - s * rotation_axis[:,1], 
            t * rotation_axis[:,1] * rotation_axis[:,2] + s * rotation_axis[:,0], 
            t * rotation_axis[:,2] * rotation_axis[:,2] + c
            ],
        dim=-1
    )
    
    rotation_matrix = torch.cat([
        rot_1.unsqueeze(1),
        rot_2.unsqueeze(1),
        rot_3.unsqueeze(1)
    ], dim=1)
    
    rotation_matrix = rotation_matrix.float().to(desired_vector.device) 
    
    return rotation_matrix



def rotation_matrix_from_vectors(input_vector, desired_vector):
    input_vector = input_vector / torch.norm(input_vector)
    desired_vector = desired_vector / torch.norm(desired_vector)
    
    rotation_axis = torch.cross(input_vector, desired_vector)
    rotation_axis = rotation_axis / torch.norm(rotation_axis)
    
    dot_product = torch.dot(input_vector, desired_vector)
    
    angle = torch.acos(dot_product)
    
    c = torch.cos(angle)
    s = torch.sin(angle)
    t = 1 - c
    
    rotation_matrix = torch.tensor([
        [t * rotation_axis[0] * rotation_axis[0] + c, t * rotation_axis[0] * rotation_axis[1] - s * rotation_axis[2], t * rotation_axis[0] * rotation_axis[2] + s * rotation_axis[1]],
        [t * rotation_axis[0] * rotation_axis[1] + s * rotation_axis[2], t * rotation_axis[1] * rotation_axis[1] + c, t * rotation_axis[1] * rotation_axis[2] - s * rotation_axis[0]],
        [t * rotation_axis[0] * rotation_axis[2] - s * rotation_axis[1], t * rotation_axis[1] * rotation_axis[2] + s * rotation_axis[0], t * rotation_axis[2] * rotation_axis[2] + c]
    ], dtype=torch.float32).to(desired_vector.device) 
    
    return rotation_matrix


def quaternion_rotation_y(angle_deg):
    angle_rad = np.radians(angle_deg)
    half_angle = angle_rad / 2
    quat = np.array([np.cos(half_angle), 0, np.sin(half_angle), 0])
    return quat

def quaternion_rotation_x(angle_deg):
    angle_rad = np.radians(angle_deg)
    half_angle = angle_rad / 2
    quat = np.array([np.cos(half_angle), np.sin(half_angle), 0, 0])
    return quat

def quaternion_rotation_z(angle_deg):
    angle_rad = np.radians(angle_deg)
    half_angle = angle_rad / 2
    quat = np.array([np.cos(half_angle), 0, 0, np.sin(half_angle)])
    return quat


def rotation_to_quaternion(rotation_matrix):
    tr = rotation_matrix.trace()
    if tr > 0:
        S = torch.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
        qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
        qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
    elif (rotation_matrix[0, 0] > rotation_matrix[1, 1]) and (rotation_matrix[0, 0] > rotation_matrix[2, 2]):
        S = torch.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
        qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
        qx = 0.25 * S
        qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / S
        qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / S
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        S = torch.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
        qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
        qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / S
        qy = 0.25 * S
        qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / S
    else:
        S = torch.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
        qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
        qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / S
        qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / S
        qz = 0.25 * S

    quaternion = torch.tensor([qw, qx, qy, qz], dtype=torch.float32).to(rotation_matrix.device)
    quaternion /= quaternion.norm()
    
    return quaternion



def batched_rotation_to_quaternion(rotation_matrices):
    batched_trace = rotation_matrices[:, 0, 0] + rotation_matrices[:, 1, 1] + rotation_matrices[:, 2, 2]
    b_size = len(rotation_matrices)
    
    qws = torch.empty(b_size, dtype=torch.float32).to(rotation_matrices.device)
    qxs = torch.empty(b_size, dtype=torch.float32).to(rotation_matrices.device)
    qys = torch.empty(b_size, dtype=torch.float32).to(rotation_matrices.device)
    qzs = torch.empty(b_size, dtype=torch.float32).to(rotation_matrices.device)
    
    # for tr > 0
    valid_part = (batched_trace > 0)
    if valid_part.sum() > 0:
        rotation_matrix = rotation_matrices[valid_part]
        S = torch.sqrt(batched_trace[valid_part] + 1.0) * 2
        qws[valid_part] = 0.25 * S
        qxs[valid_part] = (rotation_matrix[:, 2, 1] - rotation_matrix[:, 1, 2]) / S
        qys[valid_part] = (rotation_matrix[:, 0, 2] - rotation_matrix[:, 2, 0]) / S
        qzs[valid_part] = (rotation_matrix[:, 1, 0] - rotation_matrix[:, 0, 1]) / S
    
    # Where x-axis is main rotation-axis
    valid_part = (rotation_matrices[:, 0, 0] >= rotation_matrices[:, 1, 1]) * (rotation_matrices[:, 0, 0] >= rotation_matrices[:, 2, 2])
    valid_part = valid_part * (batched_trace <= 0)
    if valid_part.sum() > 0:
        rotation_matrix = rotation_matrices[valid_part]
        S = torch.sqrt(1.0 + rotation_matrix[:, 0, 0] - rotation_matrix[:, 1, 1] - rotation_matrix[:, 2, 2]) * 2
        qws[valid_part] = (rotation_matrix[:, 2, 1] - rotation_matrix[:, 1, 2]) / S
        qxs[valid_part] = 0.25 * S
        qys[valid_part] = (rotation_matrix[:, 0, 1] + rotation_matrix[:, 1, 0]) / S
        qzs[valid_part] = (rotation_matrix[:, 0, 2] + rotation_matrix[:, 2, 0]) / S
    
    # Where y-axis is main rotation-axis
    valid_part = (rotation_matrices[:, 1, 1] > rotation_matrices[:, 0, 0]) * (rotation_matrices[:, 1, 1] >= rotation_matrices[:, 2, 2])
    valid_part = valid_part * (batched_trace <= 0)
    if valid_part.sum() > 0:
        rotation_matrix = rotation_matrices[valid_part]
        S = torch.sqrt(1.0 + rotation_matrix[:, 1, 1] - rotation_matrix[:, 0, 0] - rotation_matrix[:, 2, 2]) * 2
        qws[valid_part] = (rotation_matrix[:, 0, 2] - rotation_matrix[:, 2, 0]) / S
        qxs[valid_part] = (rotation_matrix[:, 0, 1] + rotation_matrix[:, 1, 0]) / S
        qys[valid_part] = 0.25 * S
        qzs[valid_part] = (rotation_matrix[:, 1, 2] + rotation_matrix[:, 2, 1]) / S

    # Where z-axis is main rotation-axis
    valid_part = (rotation_matrices[:, 2, 2] > rotation_matrices[:, 0, 0]) * (rotation_matrices[:, 2, 2] >= rotation_matrices[:, 1, 1])
    valid_part = valid_part * (batched_trace <= 0)
    if valid_part.sum() > 0:
        rotation_matrix = rotation_matrices[valid_part]
        S = torch.sqrt(1.0 + rotation_matrix[:, 2, 2] - rotation_matrix[:, 0, 0] - rotation_matrix[:, 1, 1]) * 2
        qws[valid_part] = (rotation_matrix[:, 1, 0] - rotation_matrix[:, 0, 1]) / S
        qxs[valid_part] = (rotation_matrix[:, 0, 2] + rotation_matrix[:, 2, 0]) / S
        qys[valid_part] = (rotation_matrix[:, 1, 2] + rotation_matrix[:, 2, 1]) / S
        qzs[valid_part] = 0.25 * S

    
    quaternion = torch.stack([qws, qxs, qys, qzs], dim=-1).to(rotation_matrices.device)
    quaternion /= torch.norm(quaternion, dim=-1, keepdim=True)
    
    return quaternion



    # Ensure the input matrices are 3x3
    if rotation_matrices.size(-1) != 3 or rotation_matrices.size(-2) != 3:
        raise ValueError("Input matrices should be of size (batch_size, 3, 3)")

    # Extract the elements of the rotation matrix
    m00, m01, m02 = rotation_matrices[:, 0, 0], rotation_matrices[:, 0, 1], rotation_matrices[:, 0, 2]
    m10, m11, m12 = rotation_matrices[:, 1, 0], rotation_matrices[:, 1, 1], rotation_matrices[:, 1, 2]
    m20, m21, m22 = rotation_matrices[:, 2, 0], rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2]

    # Calculate the quaternion elements
    qw = 0.5 * torch.sqrt(1 + m00 + m11 + m22)
    qx = (m21 - m12) / (4 * qw)
    qy = (m02 - m20) / (4 * qw)
    qz = (m10 - m01) / (4 * qw)

    # Stack the quaternion elements
    quaternion = torch.stack([qw, qx, qy, qz], dim=-1)
    quaternion /= quaternion.norm(dim=-1, keepdim=True)

    return quaternion


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))




def smooth_mapping(x, zero_level = -0.2, zero_value=0.9, cut_level = -1/1.414):
    x[x>=zero_level] = 1
    # x[x>=zero_level] = (x[x>=zero_level] - zero_level) / (1-zero_level) * (1-zero_value) + zero_value
    x[x<zero_level] = ((1-(zero_level - x[x<zero_level])/(zero_level + 1))**2) * zero_value
    
    x[x<cut_level] = 0
    
    return x
        

def rot_weighting(rotations, dir_pp_normalized, return_rot_vector=False):
    # use negative z-direction as unit vector
    unit_vector = torch.tensor([0., 0., -1.,]).float().to(rotations.device)
    
    if rotations.isnan().sum() > 0:
        print("warning!")
        mask = (rotations.isnan().sum(1,2) + rotations.isinf().sum(1,2)) > 0
        do_masking = True
    else:
        do_masking = False
    R = rotations
    
    # need to invert, as I want unit-vector towards outsides
    rotated_unit_vector = R @ unit_vector
    cos_sim = -(rotated_unit_vector * dir_pp_normalized).sum(-1)
    
    # map 
    weights = smooth_mapping(cos_sim)
    
    if do_masking:
        rotated_unit_vector[mask, 0] = 0
        rotated_unit_vector[mask, 1] = 0
        rotated_unit_vector[mask, 2] = -1
        weights[mask] = 0
    
    if return_rot_vector:    
        return weights.unsqueeze(-1), rotated_unit_vector
    else:
        return weights.unsqueeze(-1)

def dilate_mask(mask, kernel_size: int=30):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = mask.copy().astype(np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    return dilated_mask

def erode_mask(mask, kernel_size: int=30):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = mask.copy().astype(np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    return eroded_mask