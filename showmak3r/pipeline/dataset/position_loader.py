import numpy as np
import cv2
import os

from pathlib import Path
from tqdm import tqdm

from showmak3r.utils.io_utils import read_pickle
from showmak3r.pipeline.dataset.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, read_intrinsics_text, read_points3D_binary, read_points3D_text, qvec2rotmat

def load_colmap_camdicts(colmap_dir: Path):
    try:
        cameras_extrinsic_file = (colmap_dir / "images.bin")
        cameras_intrinsic_file = (colmap_dir / "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = (colmap_dir / "images.txt")
        cameras_intrinsic_file = (colmap_dir / "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    cam_dicts = dict()
    cc_dicts = dict()
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
    
        uid = intr.id
        R = (qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        fname = str(os.path.basename(extr.name).split(".")[0])

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
            cx = intr.params[1]
            cy = intr.params[2]
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
        else:
            raise NotImplementedError(f"Unsupported camera model: {intr.model}")

        K = np.eye(4, dtype=np.float32)
        K[0,0] = focal_length_x
        K[1,1] = focal_length_y
        K[0,2] = cx
        K[1,2] = cy

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, -1] = T

        cam_dicts[fname] = dict(
            fname=fname,
            H = height,
            W = width,
            cx = cx,
            cy = cy,
            fx = focal_length_x,
            fy = focal_length_y,
            w2c = w2c,
            intrinsic = K,
            projection = K @ w2c
        )

        # calculate cam_center
        c2w = np.eye(4)
        c2w[:3,:3] = R.T
        c2w[:3,3] = T
        w2c = np.linalg.inv(c2w)
        cc = w2c[:3, 3]
        cc_dicts[fname] = cc

    return cam_dicts, cc_dicts


def load_sfm_points(colmap_dir: Path, cam_center_dict: dict):
    bin_path = (colmap_dir / "points3D.bin")
    txt_path = (colmap_dir / "points3D.txt")
    try:
        xyz, rgb, _, normal = read_points3D_text(txt_path, cc_dict=cam_center_dict, get_normal=True)
    except:
        xyz, rgb, _ = read_points3D_binary(bin_path)

    return xyz, rgb

def prepare_data(data_dir, model_dir):
    video_dir = data_dir / "video"
    assert video_dir.exists()
    composite_dir = data_dir / "composite"
    assert composite_dir.exists()
    stage_dir = model_dir / "stage"
    assert stage_dir.exists()
    
    # load merged result
    merged_result_path = video_dir / "merged_result.pkl"
    assert merged_result_path.exists()
    merged_result = read_pickle(merged_result_path)
    print("Loaded merged result.")
    
    # load aligned result if exists
    aligned_result_path = model_dir / "actor" / "aligned_result.pkl"
    if aligned_result_path.exists():
        aligned_result = read_pickle(aligned_result_path)
        print("Loaded aligned result.")
    else:
        aligned_result = None
        print("No aligned result found.")

    # load associated result if exists
    associated_result_path = model_dir / "actor" / "associated_result.pkl"
    if associated_result_path.exists():
        associated_result = read_pickle(associated_result_path)
        print("Loaded associated result.")
    else:
        associated_result = None
        print("No associated result found.")
    
    # load frames
    img_dir = video_dir / 'frames'
    img_dict = dict()
    for img_fname in sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
        img = cv2.imread(str(img_fname))
        fname = str(img_fname.name.split(".")[0])
        img_dict[fname] = img
    print("Loaded frames.")

    # load aligned depth maps
    depth_dir = composite_dir / "depths" / "original_aligned"
    assert depth_dir.exists()
    depth_dict = dict()
    for depth_fname in sorted(list(depth_dir.glob("*.npy"))):
        idx = str(depth_fname.name.split("_aligned")[0])
        if idx in img_dict.keys():
            depth_map = np.load(str(depth_fname))
            depth_dict[idx] = depth_map
    print("Loaded aligned depth maps.")

    # load shot detection info
    boundary_list = []
    with open(video_dir / "shot_change.log", 'r') as file:
        for line in file:
            line = line.strip()
            frame_boundary = int(line)
            boundary_list.append(frame_boundary)
    boundary_list.append(len(img_dict))
    print("Loaded shot detection info.")

    # load colmap cameras
    sfm_dir = composite_dir / "undistorted" / "sparse" / "0"
    assert sfm_dir.exists()
    cam_infos_unsorted, cam_center_dict = load_colmap_camdicts(sfm_dir)
    colmap_camdicts = {k: v for k, v in cam_infos_unsorted.copy().items() if v['fname'] in img_dict.keys()}
    print("Loaded colmap cameras.")

    # load sfm points
    xyz, rgb = load_sfm_points(sfm_dir, cam_center_dict)
    print("Loaded sfm points.")

    # load peronal masks and depth maps
    mask_dir = video_dir / 'personal_masks'
    mask_dicts = dict()
    depth_dicts = dict()
    
    for shot_id in range(len(boundary_list)):
        mask_dicts[shot_id] = dict()
        depth_dicts[shot_id] = dict()
    
    for personal_mask_dir in sorted(mask_dir.iterdir()):
        if personal_mask_dir.is_dir():
            person_masks = dict()
            person_depths = dict()
            shot_id = int(personal_mask_dir.name.split('_')[0])
            pnum = int(personal_mask_dir.name.split('_')[1])
            for mask_fname in tqdm(sorted(personal_mask_dir.glob("*.png")), desc=f"Loading individual depth - person {pnum}"):
                fname = str(mask_fname.name.split(".")[0])
                mask = cv2.imread(str(mask_fname), cv2.IMREAD_UNCHANGED)
                mask = mask / 255.0
                if (mask==0).all(): # if empty mask, skip
                    continue
                person_masks[fname] = mask
                if len(mask.shape)==3:
                    mask = mask[:, :, 0]
                person_depths[fname] = depth_dict[fname] * mask[:, :, np.newaxis]
                person_depths[fname][mask==0.] = float('inf')
            mask_dicts[shot_id][pnum] = person_masks
            depth_dicts[shot_id][pnum] = person_depths
    print("Loaded personal masks and depth maps.")

    # load stage depth maps of video frames
    stage_depth_dir = stage_dir / "depth_maps"
    stage_depth_dict = dict()
    for depth_fname in sorted(list(stage_depth_dir.glob("*.npy"))):
        fname = str(depth_fname.name.split(".")[0])
        if fname not in img_dict.keys(): # use only stage depth maps of video frames
            continue
        stage_depth_dict[fname] = np.load(str(depth_fname))
    print("Loaded stage depth maps.")

    return {
        "merged": merged_result,
        "aligned": aligned_result,
        "associated": associated_result,
        "images": img_dict,
        "masks": mask_dicts,
        "depths": depth_dicts,
        "colmap": colmap_camdicts,
        "boundary": boundary_list,
        "sfm_points": (xyz, rgb),
        "stage_depths": stage_depth_dict
    }