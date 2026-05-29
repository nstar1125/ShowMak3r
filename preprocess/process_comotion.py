import argparse
import cv2
import pickle
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List

import sys
sys.path.append("submodules/ml-comotion")
from src.comotion_demo.utils import dataloading, helper, smpl_kinematics

from preprocess.utils.image_utils import calc_overlap

def remove_occluded_people(candidates, threshold)->List[bool]:
    '''
    Compare bbox overlap and mean depth of 3D joints to remove occluded people.
    '''
    _return = [True for _ in range(len(candidates))]
    for i, cand in enumerate(candidates):
        if cand is None:
            continue
        for j, other_cand in enumerate(candidates):
            if i == j or other_cand is None:
                continue
            # compare bbox overlap
            bbox_i = cand['bbox']
            bbox_j = other_cand['bbox']
            if calc_overlap(bbox_i, bbox_j, threshold):
                # compare mean depth of 3D joints
                depth_i = cand['j3d'][:, 2].mean()
                depth_j = other_cand['j3d'][:, 2].mean()
                if depth_i > depth_j:
                    _return[i] = False
                else:
                    _return[j] = False
    return _return

def process_shot_result(result_path: Path, 
                        img_dict: dict,
                        start_fnum: int,
                        frame_num: int,
                        pnum_offset: int):
    '''
    Load and process comotion result from a single shot.
    '''
    start_fid = start_fnum - 1
    
    smpl_decoder = smpl_kinematics.SMPLKinematics()
    
    hmr_result = torch.load(result_path / "frames.pt") # dict_keys(['id', 'pose', 'trans', 'betas', 'frame_idx'])  
    tracked_pnums = hmr_result['id'].unique().tolist()

    bbox_result = []
    with open(result_path / "frames.txt", "r") as f:
        for line in f:
            bbox_result.append([float(x) for x in line.strip().split(",")])
    bbox_tensor = torch.tensor(bbox_result) # (M, 10) = (M, fid(1) + pid(1) + 4 + etc)

    track_result = dict()
    for pnum in tracked_pnums: # person num (starts from 1)
        track_result[pnum_offset+pnum] = dict()
    
    hmr_tensor = torch.concat([
        hmr_result['frame_idx'].unsqueeze(-1), 
        torch.ones((len(hmr_result['id']), 1), dtype=torch.float32), # scale
        hmr_result['trans'],
        hmr_result['pose'], # rotation + pose
        hmr_result['betas'],
    ], 1) # (M, 87) = (M, 1 + 1 + 3 + 72 + 10)

    for fid in range(frame_num):
        cur_fname = f"frame_{start_fnum + fid:04d}" # starts from frame_0001
        
        image = dataloading.convert_image_to_tensor(img_dict[cur_fname])
        K = dataloading.get_default_K(image)

        for pnum in tracked_pnums:
            select_indices = hmr_result['id']==pnum
            valid_fids = hmr_result['frame_idx'][select_indices]
            if fid not in valid_fids.tolist():
                continue
            idx = valid_fids.tolist().index(fid)

            smpl_data = hmr_tensor[select_indices][:, 1:] # (N, 86)
            bbox_data = bbox_tensor[select_indices][:, 2:6] # (N, 4)

            pred_3d = smpl_decoder(smpl_data[idx][-10:], smpl_data[idx][4:-10], smpl_data[idx][1:4])
            pred_2d = helper.project_to_2d(K, pred_3d)
            
            track_result[pnum_offset+pnum][cur_fname] = {
                "bbox": bbox_data[idx],
                "smpl_param": smpl_data[idx],
                "H": img_dict[cur_fname].shape[0],
                "W": img_dict[cur_fname].shape[1],
                "j3d": pred_3d, # (24,3)
                "j2d": pred_2d, # (24,2)
            }
        
        # get people dictionaries in current frame (None if not detected)
        candidates = []
        for pnum in tracked_pnums:
            if cur_fname in track_result[pnum_offset+pnum]:
                candidates.append(track_result[pnum_offset+pnum][cur_fname])
            else:
                candidates.append(None)
        
        # remove occluded people in current frame
        valid_list = remove_occluded_people(candidates, threshold=0.8)
        for i, is_valid in enumerate(valid_list):
            if not is_valid:
                print(f"Removing person {pnum_offset+tracked_pnums[i]} in {cur_fname} due to occlusion.")
                del track_result[pnum_offset+tracked_pnums[i]][cur_fname]

    return track_result

def main(args):
    data_dir = Path(args.data_dir)
    comotion_path = data_dir / "comotion"
    assert comotion_path.exists()

    # load images
    img_dir = data_dir / "frames"
    img_dict = dict()
    img_list = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    for img_dir in tqdm(img_list, desc="Loading images", total=len(img_list)):
        img = cv2.imread(str(img_dir))
        fname = str(img_dir.name.split(".")[0])  # ex) frame_0001
        img_dict[fname] = img

    # load shot change info
    boundary_list = [0]
    with open(data_dir / "shot_change.log", 'r') as file:
        for line in file:
            line = line.strip()
            frame_boundary = int(line)
            boundary_list.append(frame_boundary)
    boundary_list.append(len(img_dict))
    
    # change comotion format
    print("Processing comotion results ...")
    detected_pcount = 0
    shot_results = dict()
    for shot_id, shot_dir in enumerate(sorted(comotion_path.glob("shot_*"))):
        start_fnum = boundary_list[shot_id]+1
        frame_num = boundary_list[shot_id+1] - boundary_list[shot_id]
        shot_result = process_shot_result(shot_dir, img_dict, start_fnum, frame_num, detected_pcount)
        shot_results[shot_id] = shot_result # shot_id -> pnum -> fname
        detected_pcount+=len(shot_result)
    
    with open(data_dir / "comotion" / "comotion_result.pkl", "wb") as f:
        pickle.dump(shot_results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='video data path')
    args = parser.parse_args()
    main(args)