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
import math
import numpy as np
from typing import NamedTuple, List
import os
import random

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

color_fname = os.path.join(os.path.dirname(__file__), '_colors.txt')
COLOR_DICT = dict()
with open(color_fname, 'r') as f:
    colors = f.readlines()

for color in colors:
    if color[0] == '_':
        color_name = color[1:-1].lower()
        COLOR_DICT[color_name] = []
    else:
        _color = color[1:-1].lower()
        if _color.split("#")[0] in ["50"]:
            continue
        if color_name == 'grey' and _color.split("#")[0] in ["100", "200", "300"]:
            continue

        rgb_color = _color.split("#")[-1]
        assert len(rgb_color) == 6, f"invalid color name {rgb_color}"
        rgb_color = (int(rgb_color[:2], 16), int(rgb_color[2:4], 16), int(rgb_color[4:], 16))
        COLOR_DICT[color_name].append(rgb_color)

def get_color(idx=0, interval=1, get_color_lists=False, theme='green'):
    # use higher value in interval, if you want to sample every n-iters
    # "https://www.materialpalette.com/colors"

    if isinstance(theme, str):
        if theme.lower() not in COLOR_DICT:
            print(f"[WARNING] invalid theme {theme.lower()}")
            rand_idx = random.randint(0, len(COLOR_DICT)-1)
            theme = list(COLOR_DICT.keys())[rand_idx]
            print(f"[WARNING] Randomly chosed following theme {theme}")
        
        # get color
        colors = COLOR_DICT[theme]
        idx = (idx * interval) % len(colors)
        rgb_color = colors[idx]
    else:
        themes = []
        for _theme in theme:
            if _theme.lower() in COLOR_DICT:
                themes.append(_theme)
        colors = []
        for _theme in themes:
            colors.extend(COLOR_DICT[_theme])
        idx = (idx * interval) % len(colors)
        rgb_color = colors[idx]

    if get_color_lists:
        return rgb_color, colors
    else:
        return rgb_color
    
# ======


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t.reshape(-1)
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY, cx=0, cy=0):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    cx = cx * right 
    cy = cy * top 

    P = torch.zeros(4, 4)

    z_sign = 1.0

   
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left) + (2*cx) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom) + (2*cy) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)          #### Why 2. is omitted here?? (using z-direction only half?)
    P[2, 3] = -(zfar * znear) / (zfar - znear)  
    
    return P


def getCVProjectionMatrix(width, height, fovX, fovY, cx, cy):
    focalX = fov2focal(fovX, width)
    focalY = fov2focal(fovY, height)

    cx = cx * width
    cy = cy * height

    P = np.eye(3)
    P[0,0] = focalX
    P[1,1] = focalY
    P[0,2] = cx + width/2 
    P[1,2] = cy + height/2
    P[2,2] = 1

    return P





def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray):
    # Ensure that the input vectors are normalized
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    # Calculate the cross product of the two vectors
    cross_product = np.cross(vec1, vec2)
    
    # Calculate the dot product of the two vectors
    dot_product = np.dot(vec1, vec2)
    
    # Calculate the skew-symmetric matrix
    skew_symmetric_matrix = np.array([[0, -cross_product[2], cross_product[1]],
                                      [cross_product[2], 0, -cross_product[0]],
                                      [-cross_product[1], cross_product[0], 0]])
    
    # Calculate the rotation matrix using Rodrigues' rotation formula
    rotation_matrix = np.identity(3) + skew_symmetric_matrix + np.dot(skew_symmetric_matrix, skew_symmetric_matrix) * (1 / (1 + dot_product))
    
    return rotation_matrix




def project_points_to_cam(view, points, image_res=None):
    """
    based on CUDA code
    """
    # CUDA code use tranposed transformation as input (somewhat strange)
    projection_matrix = view.full_proj_transform.clone().detach().cpu().numpy().T
    
    width = view.image_width if image_res is None else image_res[1]
    height = view.image_height if image_res is None else image_res[0]
    
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    if isinstance(points, List):
        points = np.array(points)

    org_shape = points.shape[:-1]
    points = points.reshape(-1, 3)

    points = np.concatenate([points, np.ones_like(points[...,0:1])], axis=-1)
    p_cam = np.einsum('ij, bj -> bi', projection_matrix, points)
    p_cam = p_cam[...,:3] / (p_cam[..., 3:] + 1e-9)
    
    # Now p_cam points mean the points in NDC space.
    p_cam[...,0] = ((p_cam[...,0] + 1.) * width - 1.) * 0.5
    p_cam[...,1] = ((p_cam[...,1] + 1.) * height - 1.) * 0.5
    
    projected_points= p_cam[..., :2]
    projected_points = projected_points.reshape(*org_shape, 2)

    if (np.isinf(projected_points).sum() + np.isnan(projected_points).sum()) > 0:
        assert(0)
    
    return projected_points


# TODO - deprecated
# def old_project_points_to_cam(view, points, image_res=None):
#     """
#     based on CUDA code
#     """
#     w2c = view.world_view_transform.clone().detach().cpu().numpy().T

#     # c2w = np.linalg.inv(w2c)

#     # c2w[0:3, 0] *= -1
#     # c2w[0:3, 1] *= -1
#     # c2w[0:3, 2] *= -1    
    
#     # w2c = np.linalg.inv(c2w)

#     cx = getattr(view, 'cx', 0) / 2     # it has doubled center point here
#     cy = getattr(view, 'cy', 0) / 2
    
#     width = view.image_width if image_res is None else image_res[1]
#     height = view.image_height if image_res is None else image_res[0]
#     projection = getCVProjectionMatrix(width, height, view.FoVx, view.FoVy, cx, cy)
#     # Make cv2 projection here instead

#     # projection = view.projection_matrix.clone().detach().cpu().numpy()

#     if isinstance(points, torch.Tensor):
#         points = points.detach().cpu().numpy()

#     org_shape = points.shape[:-1]
#     points = points.reshape(-1, 3)

#     points = np.concatenate([points, np.ones_like(points[...,0:1])], axis=-1)
#     p_cam = np.einsum('ij, bj -> bi', w2c, points)
#     p_cam = p_cam[...,:3] / (p_cam[..., 3:] + 1e-9)
    
#     # p_cam[...,:3] /= (np.abs(p_cam[...,2:]) + 1e-9)
#     p_cam[...,2:] = np.abs(p_cam[...,2:])
    
    
#     projected_points = np.einsum('ij, bj-> bi', projection, p_cam)
#     # projected_points = projected_points[...,:2]

#     # projected_points = np.einsum('ij, btnj-> btni', gl2cv, projected_points)  
#     projected_points = projected_points[...,:2] / np.abs((projected_points[..., -1:] + 1e-9))       # To handle points pair behind camera, need to add abs()

#     projected_points = projected_points.reshape(*org_shape, 2)
#     return projected_points
