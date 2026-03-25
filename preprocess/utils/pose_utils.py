import numpy as np
import torch

def smpl_to_op25(smpl_joints): # SMPL 24 joints -> OpenPose 25 joints
    """
    Convert SMPL 24 joints to OpenPose 25 joints format
    
    SMPL joints (24):
    0: pelvis, 1: left_hip, 2: right_hip, 3: spine1, 4: left_knee, 5: right_knee, 
    6: spine2, 7: left_ankle, 8: right_ankle, 9: spine3, 10: left_foot, 11: right_foot, 
    12: neck, 13: left_collar, 14: right_collar, 15: head, 16: left_shoulder, 17: right_shoulder, 
    18: left_elbow, 19: right_elbow, 20: left_wrist, 21: right_wrist, 22: left_hand, 23: right_hand
    
    OpenPose 25 joints:
    0: nose, 1: neck, 2: right_shoulder, 3: right_elbow, 4: right_wrist, 5: left_shoulder,
    6: left_elbow, 7: left_wrist, 8: mid_hip, 9: right_hip, 10: right_knee, 11: right_ankle,
    12: left_hip, 13: left_knee, 14: left_ankle, 15: right_eye, 16: left_eye, 17: right_ear,
    18: left_ear, 19: left_big_toe, 20: left_small_toe, 21: left_heel, 22: right_big_toe,
    23: right_small_toe, 24: right_heel
    """
    
    # Initialize with invalid joints
    op25_joints = [torch.tensor([np.nan, np.nan], dtype=torch.float32) for _ in range(25)]
    
    # SMPL to OpenPose mapping
    smpl_to_op_mapping = {
        15: 0,  # head -> nose (approximation)
        12: 1,  # neck -> neck
        17: 2,  # right_shoulder -> right_shoulder
        19: 3,  # right_elbow -> right_elbow
        21: 4,  # right_wrist -> right_wrist
        16: 5,  # left_shoulder -> left_shoulder
        18: 6,  # left_elbow -> left_elbow
        20: 7,  # left_wrist -> left_wrist
        0: 8,   # pelvis -> mid_hip
        2: 9,   # right_hip -> right_hip
        5: 10,  # right_knee -> right_knee
        8: 11,  # right_ankle -> right_ankle
        1: 12,  # left_hip -> left_hip
        4: 13,  # left_knee -> left_knee
        7: 14,  # left_ankle -> left_ankle
        # 15-18: eyes and ears (not available in SMPL)
        10: 19, # left_foot -> left_big_toe (approximation)
        # 20: left_small_toe (not available)
        10: 21, # left_foot -> left_heel (approximation)
        11: 22, # right_foot -> right_big_toe (approximation)
        # 23: right_small_toe (not available)
        11: 24, # right_foot -> right_heel (approximation)
    }
    
    # Apply mapping
    for smpl_idx, op_idx in smpl_to_op_mapping.items():
        if smpl_idx < len(smpl_joints):
            op25_joints[op_idx] = smpl_joints[smpl_idx]
    
    return op25_joints

def dwpose_to_op25(dw_joints): 
    """
    dwpose is simple op18(19) + foots / pelvis omitted / legs are swapped
    """
    joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 19, 20, 21], dtype=np.int32) # skip 8
    op25_joints = [torch.tensor([np.nan, np.nan, np.nan], dtype=torch.float32) for _ in range(25)]
    for op_idx, j_idx in enumerate(joint_indices):
        op25_joints[j_idx] = dw_joints[op_idx] # dw joints are sequentially mapped into op25_joints 

        l_hip = dw_joints[11]
        r_hip = dw_joints[8]
        if ~l_hip.isnan().all() and ~r_hip.isnan().all(): # both not nan
            op25_joints[8] = (l_hip + r_hip) / 2.
    return op25_joints