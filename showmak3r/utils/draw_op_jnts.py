import numpy as np
import math
import cv2
import matplotlib
from PIL import Image
from typing import List

import warnings
# Ignore specific UserWarning
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_5m_224 in registry with controlnet_aux.segment_anything.modeling.tiny_vit_sam.tiny_vit_5m_224.*")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_11m_224 in registry with controlnet_aux.segment_anything.modeling.tiny_vit_sam.tiny_vit_11m_224.*")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_21m_224 in registry with controlnet_aux.segment_anything.modeling.tiny_vit_sam.tiny_vit_21m_224.*")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_21m_384 in registry with controlnet_aux.segment_anything.modeling.tiny_vit_sam.tiny_vit_21m_384.*")
warnings.filterwarnings("ignore", message="Overwriting tiny_vit_21m_512 in registry with controlnet_aux.segment_anything.modeling.tiny_vit_sam.tiny_vit_21m_512.*")


from controlnet_aux.open_pose import draw_poses, PoseResult
from controlnet_aux.open_pose.body import Keypoint, BodyResult
from controlnet_aux.util import HWC3


def op25_to_op18(op_jnts):
    j_inds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=np.int32)

    op_joints = []
    for j_ind in j_inds:
        if j_ind >= len(op_jnts):
            op_joints.append(None)
        else:
            op_joints.append(op_jnts[j_ind])

    return op_joints


def op18_to_op25(op_jnts):
    j_inds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=np.int32)  # skipping 8

    op_joints = [None for _ in range(25)]
    for op_ind, j_ind in enumerate(j_inds):
        op_joints[j_ind] = op_jnts[op_ind]

    return op_joints
            

def dwpose_to_op25(op_jnts):
    """
    dwpose is simple op18(19) + foots, so pelvis is omitted 
    + also legs are swapped!!!!!
    """
    j_inds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 19, 20, 21,], dtype=np.int32)  # skipping 8

    op_joints = [None for _ in range(25)]
    for op_ind, j_ind in enumerate(j_inds):
        op_joints[j_ind] = op_jnts[op_ind]

    left_hip = op_jnts[11]
    right_hip = op_jnts[8]
    if left_hip is not None and right_hip is not None:
        op_joints[8] = []
        for l, r in zip(left_hip, right_hip):
            op_joints[8].append((l+r)/2.)

    return op_joints


def smpl_joints2op_joints(smpl_joints, openpose_format='coco19'):
    j_inds = smpl2op(model_type="smpl", openpose_format=openpose_format)

    op_joints = []
    for j_ind in j_inds:
        if j_ind >= len(smpl_joints):
            op_joints.append(None)
        else:
            op_joints.append(smpl_joints[j_ind])

    return op_joints
    

def smpl2op(model_type='smpl', use_hands=False, 
                          use_face=False, use_face_contour=False, 
                          openpose_format='coco19'):
    ''' Returns the indices of the permutation that maps OpenPose to SMPL
        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'
    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            # Vid2Avatar default setting
            # return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
            #                  7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
            #                 dtype=np.int32)
            
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 32, 33, 34, 29, 30, 31],        # Severe problem!!!!!!!!!
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]

            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smpl_neutral':
            return np.array([14, 12, 8, 7, 6, 9, 10, 11, 2, 1, 0, 3, 4, 5,  16, 15,18, 17,],
                            dtype=np.int32)

        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'h36':
        if model_type == 'smpl':
            return np.array([2,5,8,1,4,7,12,24,16,18,20,17,19,21],dtype=np.int32)
        elif model_type == 'smpl_neutral':
            #return np.array([2,1,0,3,4,5,12,13,9,10,11,8,7,6], dtype=np.int32)
            return [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))


def draw_op_img(op_joints, image_res, output_type='pil'):
    H = image_res if isinstance(image_res, int) else image_res[0]
    W = image_res if isinstance(image_res, int) else image_res[1]
    

    poses = PoseResult(
        BodyResult(
            keypoints=[
                Keypoint(
                    x=keypoint[0] / float(W),
                    y=keypoint[1] / float(H)
                ) if keypoint is not None else None
                for keypoint in op_joints
            ], 
            total_score=1,  # Meaningless parts
            total_parts=1,  # Meaningless parts
        ),
        left_hand=None,
        right_hand=None,
        face=None
    )
    
    canvas = draw_poses([poses], H, W, draw_body=True, draw_hand=False, draw_face=False) 
    
    detected_map = HWC3(canvas)
    if output_type == "pil":
        detected_map = Image.fromarray(detected_map)
    
    return detected_map



def draw_bodypose_with_color(canvas: np.ndarray, op_joints: List[Keypoint], image_res = None, kpts_color=(0, 0, 255), mode='op18') -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    H, W, C = canvas.shape
    if image_res is not None:
        H = image_res if isinstance(image_res, int) else image_res[0]
        W = image_res if isinstance(image_res, int) else image_res[1]
    
    poses = PoseResult(
        BodyResult(
            keypoints=[
                Keypoint(
                    x=keypoint[0] / float(W),
                    y=keypoint[1] / float(H)
                ) if keypoint is not None else None
                for keypoint in op_joints
            ], 
            total_score=1,  # Meaningless parts
            total_parts=1,  # Meaningless parts
        ),
        left_hand=None,
        right_hand=None,
        face=None
    )
    keypoints = poses.body.keypoints


    
    stickwidth = 4

    if mode == 'op18' or mode == 'op19':
        limbSeq = [
            [2, 3], [2, 6], [3, 4], [4, 5], 
            [6, 7], [7, 8], [2, 9], [9, 10], 
            [10, 11], [2, 12], [12, 13], [13, 14], 
            [2, 1], [1, 15], [15, 17], [1, 16], [16, 18],
        ]
    elif mode == 'op25':
        limbSeq = [
            [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8],     # Arms
            [2, 9],                                                 # Neck <-> Pelvis
            [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],  # Legs
            [2, 1], [1, 16], [16, 18], [1, 17], [17, 19],               # Face Joints
            [12, 20], [12, 21], [12, 22],                               # Right feet
            [15, 23], [15, 24], [15, 25],                               # Left feet
        ]
    else:
        raise NotImplementedError(f"Invalid limb type {mode}")


    for (k1_index, k2_index) in limbSeq:
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1.x, keypoint2.x]) * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in kpts_color])

    for keypoint in keypoints:
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, kpts_color, thickness=-1)

    return canvas



def draw_handpose(canvas, hands):   
    # H, W, C = canvas.shape
    # We already hold values in pixel locations
    W = 1
    H = 1
    eps = 0.01


    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    
    # (person_number*2, 21, 2)
    for i in range(len(hands)):
        peaks = hands[i]
        # peaks = np.array(peaks)
        
        for ie, e in enumerate(edges):
            # Get peaks as following
            p1 = peaks[e[0]]
            p2 = peaks[e[1]]

            if p1 is None or p2 is None:
                continue

            x1, y1 = p1[:2]
            x2, y2 = p2[:2]
            
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)

        for _, keypoint in enumerate(peaks):
            if keypoint is None:
                continue

            x, y = keypoint[:2]

            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas, faces):
    # H, W, C = canvas.shape
    # We already hold values in pixel locations
    W = 1
    H = 1
    eps = 0.01


    for lmks in faces:
        for lmk in lmks:
            if lmk is None:
                continue
            x, y = lmk[:2]
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas