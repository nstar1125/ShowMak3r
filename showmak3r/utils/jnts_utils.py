import torch
import numpy as np


def extract_square_bbox(joint_coordinates, offset_ratio=0.2, get_square=False):
    # Assuming joint_coordinates is a numpy array of shape (num_joints, 2)
    
    # Calculate the bounding box
    min_x = np.min(joint_coordinates[:, 0])
    max_x = np.max(joint_coordinates[:, 0])
    min_y = np.min(joint_coordinates[:, 1])
    max_y = np.max(joint_coordinates[:, 1])

    # Calculate width and height of the bounding box
    width = max_x - min_x
    height = max_y - min_y

    if get_square:
        width = width if width > height else height
        width = width * (1 + offset_ratio)

        center_x = (max_x + min_x) / 2
        center_y = (max_y + min_y) / 2
        
        min_x = center_x - width / 2
        max_x = center_x + width / 2
        min_y = center_y - width / 2
        max_y = center_y + width / 2

    else:
        # Calculate the offset
        offset_x = width * offset_ratio / 2
        offset_y = height * offset_ratio / 2

        # Apply the offset to the bounding box
        min_x -= offset_x
        max_x += offset_x
        min_y -= offset_y
        max_y += offset_y

    # Ensure the bounding box is within the image bounds (optional)
    # min_x = max(min_x, 0)
    # max_x = min(max_x, image_width)
    # min_y = max(min_y, 0)
    # max_y = min(max_y, image_height)

    # Return the bounding box coordinates
    bbox = [min_x, min_y, max_x, max_y]
    return bbox


def filter_invisible_joints(op_joints):
    """
    op_joints: OpenPose joints in imae sapce [B, 2]. in list format.
    
    return:
    - filter out invisible joints (with None)
    """
    if isinstance(op_joints, np.ndarray):
        op_joints = op_joints.tolist()

    left_eye = op_joints[-3]
    right_eye = op_joints[-4]
    left_ear = op_joints[-1]
    right_ear = op_joints[-2]
    nose = op_joints[0]


    if (left_eye[0] >= nose[0]) and (right_eye[0] <= nose[0]):
        # it's always frontal 
        view = "front"

    elif (left_eye[0] < nose[0]) and (right_eye[0] > nose[0]):
        # it's always back
        view = "back"
        op_joints[-4] = None
        op_joints[-3] = None
        op_joints[-2] = None
        op_joints[-1] = None
    
    elif (nose[0] > right_eye[0]):
        # it automatically include condition that nose[0] > left_eye[0]
        view = "right_side"

        # Turn on left side components
        op_joints[-3] = None
        op_joints[-1] = None
    else:
        view = "left_side"

        # Turn off right side components
        op_joints[-4] = None
        op_joints[-2] = None



    return op_joints

