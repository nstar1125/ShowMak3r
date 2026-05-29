import numpy as np
import cv2
import torch
import math
from typing import NamedTuple, List

def estimate_translation_cv2(joints_3d, joints_2d, focal_length=600, img_size=np.array([512.,512.]), proj_mat=None, cam_dist=None):
    '''
    Given 3D joints, 2D joints and camera intrinsics, estimate the camera extrinsics

    Args: joints 3d, joints 2d, focal_length, img_size, proj_mat, cam_distortion
    
    Output: translation, rotation
    '''
    if proj_mat is None:
        camK = np.eye(3)
        camK[0,0], camK[1,1] = focal_length, focal_length
        camK[:2,2] = img_size//2
    else:
        camK = proj_mat
    _, rvec, tvec, inliers = cv2.solvePnPRansac(joints_3d, joints_2d, camK, cam_dist,\
                                            flags=cv2.SOLVEPNP_EPNP,reprojectionError=30,iterationsCount=2000)
    if inliers is None:
        INVALID_TRANS=np.ones(3)*-1
        return INVALID_TRANS
    else:
        rotation_pred = rvec[:, 0]
        translation_pred = tvec[:,0]
        return translation_pred, rotation_pred

def getWorld2View(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
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