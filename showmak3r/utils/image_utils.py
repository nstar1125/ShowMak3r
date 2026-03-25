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
import os
import cv2
import subprocess
import matplotlib
import torch
import numpy as np
from math import floor
from typing import Union, Dict, List
from pathlib import Path
import math

from PIL import Image
from controlnet_aux.open_pose import draw_poses, PoseResult
from controlnet_aux.open_pose.body import Keypoint, BodyResult
from controlnet_aux.util import HWC3

def draw_bbox_w_pid(img, bbox, bgr=(0, 255, 0), thickness=2, pid=None):
    x = int(bbox[0])
    y = int(bbox[1])
    bottom_right_x = int(bbox[0] + bbox[2])
    bottom_right_y = int(bbox[1] + bbox[3])

    img = img.astype(np.uint8)
    cv2.rectangle(img, (x, y), (bottom_right_x, bottom_right_y), bgr, thickness)

    if pid is not None:
        if isinstance(pid, int):
            pid = str(pid)
        text_width, text_height = cv2.getTextSize(pid, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)[0]
        cv2.putText(img, pid, (x, y + text_height), 
            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255), 4, 2)

    return img

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

# -----------------------------------------------------------


def load_img_dict(img_dir: Path):
    # Load Images
    img_dict = dict()
    for img_fname in (list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
        img = cv2.imread(str(img_fname))
        fid = str(img_fname.name.split(".")[0])
        img_dict[fid] = img

    return img_dict


def get_error_map(rgb: torch.Tensor, rgb_gt: torch.Tensor):
    """
    we assume the format (B, C, H, W) or (C, H, W) of tesnro
    """
    errmap = (rgb - rgb_gt).square().squeeze().sum(0).sqrt().cpu().numpy() / np.sqrt(3)
    errmap = cv2.applyColorMap((errmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return errmap   

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def draw_jnts_order(img, img_jnts):
    for i, jnt in enumerate(img_jnts):
        pid = str(i)
        # write pid on top of bbox
        x = int(jnt[0])
        y = int(jnt[1])
        text_width, text_height = cv2.getTextSize(pid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(img, pid, 
            (x, y + text_height), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6,
            (191,38,211),
            1,
            2)
            
    return img


def draw_bbox(img, bbox, rgb=(0, 255, 0), thickness=2, pid=None):
    x = int(bbox[0])
    y = int(bbox[1])
    bottom_right_x = int(bbox[0] + bbox[2])
    bottom_right_y = int(bbox[1] + bbox[3])

    # Draw a rectangle on the image
    # Parameters: image, top-left corner, bottom-right corner, color (BGR), thickness
    img = img.astype(np.uint8)
    cv2.rectangle(img, (x, y), (bottom_right_x, bottom_right_y), rgb, thickness)

    if pid is not None:
        if isinstance(pid, int):
            pid = str(pid)
        # write pid on top of bbox
        text_width, text_height = cv2.getTextSize(pid, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)[0]
        cv2.putText(img, pid, 
            (x, y + text_height), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            2.0,
            (255,255,255),
            4,
            2)
            


    return img

def gen_videos(
                save_paths: Union[Dict, List, Path, str], 
                is_jpg: bool=True, 
                fps: int=10, 
                rm_dir: bool=False,
                regex_fname: str="", 
                save_tag: str="") -> None:
    """get dir-lists as dictionary or list, and make videos with file below it"""
    if isinstance(save_paths, List):
        spaths = save_paths
    elif isinstance(save_paths, Path) or isinstance(save_paths, str):
        spaths = [save_paths]
    else:
        spaths = list(save_paths.values())
    # make videos
    for spath in spaths:
        if isinstance(spath, str):
            spath = Path(spath)

        if len(regex_fname) > 0:
            # using specific expressions?
            path = spath/regex_fname
        elif is_jpg:
            path = spath/"*.jpg"
        else:
            path = spath/"*.png"

        
        spath = str(spath)
        VIDEO_NAME = spath + save_tag + ".mp4"

        cmd = f'ffmpeg -nostdin -y -f image2 -framerate {str(int(fps))} -pattern_type glob -i "{str(path)}" -pix_fmt yuv420p -c:v libx264 -x264opts keyint=25:min-keyint=25:scenecut=-1 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {VIDEO_NAME} -loglevel quiet'
        print(f"[INFO] generate video with '{cmd}'")
        os.system(cmd)
        
        # with open(os.devnull, 'wb') as devnull:
        #     subprocess.check_call(cmd.split(" "), stdout=devnull, stderr=subprocess.STDOUT)

        if rm_dir:
            cmd = f"rm -r {str(spath)}"
            os.system(cmd)



def get_colors(n_points):
    # make colors first
    hsvs = np.stack(
        [np.arange(0, n_points, dtype=np.int8) * 7,
        np.ones(n_points, dtype=np.uint8) * 255,
        np.ones(n_points, dtype=np.uint8) * 255],
        axis=-1
    )
    hsvs = hsvs.astype(np.uint8)
    bgrs = cv2.cvtColor(hsvs[None], cv2.COLOR_HSV2BGR)
    bgrs = bgrs[0]
    return bgrs



def depth2rgb(depth, colormap_name="magma_r"):
    min_value, max_value = depth.min(), depth.max()
    normalized_values = (depth - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component 
    return colors * 255


def tensor2cv(image):
    if len(image.shape) == 2:
        # mask case
        image = image[None]
    # Convert PyTorch tensor to NumPy array
    numpy_array = image.permute(1, 2, 0).cpu().numpy()  # Assuming CHW format, you may need to adjust

    # Scale the values to [0, 255] (assuming tensor values are in [-1, 1] or [0, 1])
    numpy_array = (numpy_array * 255).astype(np.uint8)

    # Convert NumPy array to OpenCV image (BGR format)
    opencv_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
    return opencv_image



def img_add_text(img:np.ndarray, input_texts, fontScale=0.6, fontColor = (191,38,211)) -> np.ndarray:
    # Write some Text
    H, W, _ = img.shape
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    thickness              = 3
    lineType               = 2
    
    if isinstance(input_texts, str) or len(input_texts) == 1:
        text_width, text_height = cv2.getTextSize(input_texts, font, fontScale, thickness)[0]
        bottomLeftCornerOfText = (10,10+text_height)
        y = 10 + text_height

        img = img.copy()
        words = input_texts.split()
        line = ''
        
        for word in words:
            test_line = line + ' ' + word if line else word
            text_size = cv2.getTextSize(test_line, font, fontScale, thickness)[0]

            if text_size[0] > (W-20):
                cv2.putText(img, line, 
                    (10, y), 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
                y += text_height + 10
                line = word
            else:
                line = test_line

        # Draw the last line
        cv2.putText(img, line, 
            (10, y), 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
            
        
        # print(text_width, text_height)

        #Save image
        #cv2.imwrite("out.jpg", img)

    elif isinstance(input_texts, list) and len(input_texts) > 1:
        # stack texts
        base = (10,10)
        line_height_ratio = 1.5
        for i, text in enumerate(input_texts):
            if isinstance(fontColor[i], list) or isinstance(fontColor[i], tuple):
                _fontColor = fontColor[i]
            else:
                _fontColor = fontColor

            text_width, text_height = cv2.getTextSize(text, font, fontScale, thickness)[0]
            bottomLeftCornerOfText = (base[0]+0, base[1]+text_height)

            cv2.putText(img, text, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                _fontColor,
                thickness,
                lineType)
            
            base = (base[0], base[1]+int(text_height*line_height_ratio))

    return img



def stitch_outputs(images: List):
    """
    stitch the outputs (it can handle both images & list of images)
    inputs:
    - images: input image lists of Dict (key: name, value: images)

    """
    # 1. check the input array scale
    n_col = len(images)
    n_row = max([len(ic) for ic in images])

    frames = []
    # For first round, check N_frames
    frame_idx = 0
    max_frame = 0
    cols = []
    for img_column in images:
        img_names = list(img_column.keys())
        single_col = []
        for img_name in img_names:
            img = img_column[img_name]
            # for first round, check whether it's video or not
            if isinstance(img, list) or len(img.shape) > 3:
                if max_frame < len(img):
                    max_frame = len(img)
                img = img[frame_idx]

            # check img dimension
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if img.max() <= 1.0:
                img = img * 255.

            img = img_add_text(img, img_name)
            single_col.append(img)
        if len(single_col) < n_row:
            for _ in range(n_row - len(single_col)):
                single_col.append(np.zeros_like(img))
        cols.append(np.concatenate(single_col, axis=0))
    final_img = np.concatenate(cols, axis=1)
    frames.append(final_img)

    # second round, now add images
    for frame_idx in range(1, max_frame):
        cols = []
        for img_column in images:
            img_names = list(img_column.keys())
            single_col = []
            for img_name in img_names:
                img = img_column[img_name]
                # for first round, check whether it's video or not
                if isinstance(img, list) or len(img.shape) > 3:
                    if frame_idx < len(img):
                        img = img[frame_idx]
                    else:
                        # if one's video is short. plot the last frame instead
                        img = img[-1]
                if img_name == "pifu_visibility":
                    print("1")
                if img_name == "pifu_normal" and frame_idx == 15:
                    print("1")

                if img[...,0:3].max() <= 10.0: #### TODO: I don't know why, but opengl render results often exceed 1.0 
                    img[img>1.0] = 1.0
                    img[img<0.0] = 0.0
                    img = img * 255

                # check img dimension
                if img.shape[-1] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                img = img_add_text(img, img_name)
                single_col.append(img)
            if len(single_col) < n_row:
                for _ in range(n_row - len(single_col)):
                    single_col.append(np.zeros_like(img))
            cols.append(np.concatenate(single_col, axis=0))
        final_img = np.concatenate(cols, axis=1)
        frames.append(final_img)

    return frames






def get_crop_img_w_jnts(img, bbox, projected_jnts, rescale: float=1.2, resize: int=512):
    min_x = bbox[0]
    min_y = bbox[1]
    max_x = min_x + bbox[2]
    max_y = min_y + bbox[3]
    
    _w = int((max_x-min_x)*rescale)
    _h = int((max_y-min_y)*rescale)
    c_x = (min_x + max_x) // 2
    c_y = (min_y + max_y) // 2
    
    w = _w if _w>_h else _h
    h = w

    x = floor(c_x - w//2)
    y = floor(c_y - h//2)

    '''Crop in rectangular shape'''
    '''pad imgs when bbox is out of img'''
    x_front = 0   # offset for the case when we padded in front of the img.
    y_front = 0
    x_back = 0
    y_back = 0
    
    if x<0:
        x_front = -x
    if y<0:
        y_front = -y
    if x+w>= img.shape[1]:
        x_back = x+w-img.shape[1]+1
    if y+h>=img.shape[0]:
        y_back = y+w-img.shape[0]+1

    if x_front+y_front+x_back+y_back > 0:
        ext_img = cv2.copyMakeBorder(img, y_front, y_back, x_front, x_back, cv2.BORDER_CONSTANT, value=(0,0,0))
        x = x + x_front
        y = y + y_front
    else:
        ext_img = img
    cropped_img = ext_img[y:y+h, x:x+h]
    projected_jnts = projected_jnts - np.array([[x - x_front, y - y_front]])


    if resize > 0:
        re_cropped_img = cv2.resize(cropped_img, (resize, resize))
        scale_factor = resize / h
        re_projected_jnts = (projected_jnts - np.array([[h/2, h/2]])) * scale_factor + np.array([[resize/2, resize/2]])
    
        return cropped_img, projected_jnts, re_cropped_img, re_projected_jnts
    else:
        return cropped_img, projected_jnts



def get_crop_img(img, bbox, rescale: float=1.2, resize: int=512, get_new_bbox=False):
    min_x = bbox[0]
    min_y = bbox[1]
    max_x = min_x + bbox[2]
    max_y = min_y + bbox[3]
    
    _w = int((max_x-min_x)*rescale)
    _h = int((max_y-min_y)*rescale)
    c_x = (min_x + max_x) // 2
    c_y = (min_y + max_y) // 2
    
    w = _w if _w>_h else _h
    h = w

    x = floor(c_x - w//2)
    y = floor(c_y - h//2)

    '''Crop in rectangular shape'''
    '''pad imgs when bbox is out of img'''
    x_front = 0   # offset for the case when we padded in front of the img.
    y_front = 0
    x_back = 0
    y_back = 0
    
    if x<0:
        x_front = -x
    if y<0:
        y_front = -y
    if x+w>= img.shape[1]:
        x_back = x+w-img.shape[1]+1
    if y+h>=img.shape[0]:
        y_back = y+w-img.shape[0]+1

    if x_front+y_front+x_back+y_back > 0:
        ext_img = cv2.copyMakeBorder(img, y_front, y_back, x_front, x_back, cv2.BORDER_CONSTANT, value=(0,0,0))
        x = x + x_front
        y = y + y_front
    else:
        ext_img = img
    cropped_img = ext_img[y:y+h, x:x+h]

    if resize > 0:
        cropped_img = cv2.resize(cropped_img, (resize, resize))
    
    if get_new_bbox:
        if y_front > 0:
            y = -y_front
        if x_front > 0:
            x = -x_front
        new_bbox = [x, y, h, h]

        return cropped_img, new_bbox
    else:
        return cropped_img
            


def padded_resize(image, target_size):
    h, w = image.shape[:2]

    if h > w:
        top = 0 
        bottom = 0
        left = (h-w) // 2
        right = (h-w) // 2

        if left + right < (h-w):
            right += 1

    elif h < w:
        top = (w-h) // 2
        bottom = (w-h) // 2
        left = 0
        right = 0

        if top + bottom < (w-h):
            top += 1
    else:
        top = 0
        bottom = 0
        left = 0
        right = 0

    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    image = cv2.resize(image, target_size)

    return image