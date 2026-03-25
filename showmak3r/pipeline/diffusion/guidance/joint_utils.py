from typing import List

import torch
import numpy as np
from showmak3r.utils.camera_utils import project_points_to_cam

def filter_invisible_face_joints_w_prompts(op_joints=None, scene_cam=None, image_res=None, pj_jnts=None, is_left=False):
    """
    - op_joints: [J, 3]
    """
    if not (op_joints is None) and not (scene_cam is None):
        pj_jnts = project_points_to_cam(scene_cam, op_joints, image_res=image_res)       # [J, 2]
        
        # Check whether left side or right side
        w2c = scene_cam.world_view_transform.clone().detach().cpu().numpy().T
        points = np.array(op_joints).copy().reshape(-1, 3)
        points = np.concatenate([points, np.ones_like(points[...,0:1])], axis=-1)
        p_cam = np.einsum('ij, bj -> bi', w2c, points)
        p_cam = p_cam[...,:3] / (p_cam[..., 3:] + 1e-9)

        
        #    Compare depth (smaller z : nearer to camera)
        is_left = True if p_cam[-1, 2] < p_cam[-2, 2] else False
        
    
    # Get each joints' coordinates
    # pj_jnts = pj_jnts.tolist()
    left_eye = pj_jnts[-3]
    right_eye = pj_jnts[-4]
    left_ear = pj_jnts[-1]
    right_ear = pj_jnts[-2]
    nose = pj_jnts[0]
    
    # # Parallelize in ear-ear axis
    # ear_ear_axis = left_ear - right_ear
    # ear_ear_axis = ear_ear_axis / np.sqrt((ear_ear_axis ** 2).sum())
    
    # left_eye_loc = (left_eye * ear_ear_axis).sum()
    # right_eye_loc = (right_eye * ear_ear_axis).sum()
    # left_ear_loc = (left_ear * ear_ear_axis).sum()
    # right_ear_loc = (right_ear * ear_ear_axis).sum()
    # nose_loc = (nose * ear_ear_axis).sum()

    left_eye_loc = left_eye[0]
    right_eye_loc = right_eye[0]
    left_ear_loc = left_ear[0]
    right_ear_loc = right_ear[0]
    nose_loc = nose[0]
    
    
    # print(f"left eye: {left_eye_loc}")
    # print(f"right eye: {right_eye_loc}")
    # print(f"left ear: {left_ear_loc}")
    # print(f"right ear: {right_ear_loc}")
    # print(f"nose: {nose_loc}")
    # print(f"is_left: {is_left}")

    if isinstance(op_joints, np.ndarray):
        op_joints = op_joints.tolist()
    
    if is_left:
        if left_eye_loc > left_ear_loc:
            # it's always back
            view = "back"
            op_joints[-4] = None
            op_joints[-3] = None
            op_joints[-2] = None
            op_joints[-1] = None
            
        elif left_eye_loc > nose_loc:
            if right_eye_loc < nose_loc:
                # it's always frontal 
                view = "front side of face"
            else:
                if left_eye_loc > right_eye_loc:
                    view = "left side of face"
                    op_joints[-2] = None
                else:
                    view = "left side of head"
                    op_joints[-4] = None
                    op_joints[-2] = None
        else:
            # left_eye < nose
            view = "left side of head"
            op_joints[-4] = None
            op_joints[-2] = None
            
                    
    else:
        if right_eye_loc < right_ear_loc:
            # it's always back
            view = "back side of head"
            op_joints[-4] = None
            op_joints[-3] = None
            op_joints[-2] = None
            op_joints[-1] = None
            
        elif right_eye_loc < nose_loc:
            if left_eye_loc > nose_loc:
                # it's always frontal 
                view = "front side of face"
            else:
                if left_eye_loc > right_eye_loc:
                    view = "right side of face"
                    op_joints[-1] = None
                else:
                    view = "right side of head"
                    op_joints[-3] = None
                    op_joints[-1] = None
        else:
            # right eye > nose
            view = "right side of head"
            op_joints[-3] = None
            op_joints[-1] = None
            
            
    return op_joints, view


def get_view_prompt_of_body(op_joints, scene_cam, is_lower_body: bool=False):
    """
    - op_joints: [J, 3]
    - camera: Camera() in gtu.dataset.scene.camera
    """
    
    # Get hips names & lists
    if is_lower_body:
        left_idx = 12        # left hip
        right_idx = 9        # right hip
    else:
        left_idx = 5        # left shoulder
        right_idx = 2       # right shoulder
    
    # Check whether left side or right side
    w2c = scene_cam.world_view_transform.clone().detach().cpu().numpy().T
    points = np.array(op_joints).copy().reshape(-1, 3)
    points = np.concatenate([points, np.ones_like(points[...,0:1])], axis=-1)
    p_cam = np.einsum('ij, bj -> bi', w2c, points)
    p_cam = p_cam[...,:3] / (p_cam[..., 3:] + 1e-9)
    
    
    # Get axis_direction 
    # cam_optical_axis = np.array([0, 0, 1])
    right_to_left_axis = p_cam[left_idx] - p_cam[right_idx]
    right_to_left_axis = right_to_left_axis / np.sqrt((right_to_left_axis ** 2).sum())
    inner_product = right_to_left_axis[-1]
    x_axis = right_to_left_axis[0]


    if inner_product > 1 / np.sqrt(2) or inner_product < -1 / np.sqrt(2):
        view = "side"
    elif x_axis > 0:
        view = "front"
    else:
        view = "back"
    
    return view


def combine_prompts(head_prompts, upper_body_prompts, lower_body_prompts, projected_jnts, image_res):
    H, W = image_res[:2]
    
    # Check projected OpenPose Joints has all openpose joints
    left_eye = projected_jnts[-3]
    right_eye = projected_jnts[-4]
    left_ear = projected_jnts[-1]
    right_ear = projected_jnts[-2]
    nose = projected_jnts[0]
    jnts = [left_eye, right_eye, left_ear, right_ear, nose]
    
    head_included = True
    for _jnt in jnts:
        if (_jnt[0] < 0 or _jnt[0] > W) or (_jnt[1] < 0 or _jnt[1] > H):
            head_included = False
            
    # Check upper body is visible  (if one of joint is visible, OK)
    left_shoulder = projected_jnts[5]
    right_shoulder = projected_jnts[2]
    jnts = [left_shoulder, right_shoulder]
    
    upper_body_included = False
    for _jnt in jnts:
        if (_jnt[0] >= 0 and _jnt[0] < W) and (_jnt[1] >= 0 and _jnt[1] < H):
            upper_body_included = True
    
    
    # Check lower body is visible  (if one of joint is visible, OK)
    left_hip = projected_jnts[12]
    right_hip = projected_jnts[9]
    jnts = [left_hip, right_hip]
    
    lower_body_included = False
    for _jnt in jnts:
        if (_jnt[0] >= 0 and _jnt[0] < W) and (_jnt[1] >= 0 and _jnt[1] < H):
            lower_body_included = True
            
    
    # Finally complete the sentences
    prompts = ""
    if head_included:
        prompts += ", " + head_prompts
    if upper_body_included:
        prompts += ", " + upper_body_prompts + " view of upper body"
    if lower_body_included:
        prompts += ", " + lower_body_prompts + " view of lower body"
    
    return prompts
    