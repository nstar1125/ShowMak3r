import argparse
from pathlib import Path
import pickle
import torch
import numpy as np
from preprocess.utils.pose_utils import smpl_to_op25, dwpose_to_op25

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="merge tracklets.")
    parser.add_argument('--data_dir', type=str, help='path to data')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # load 3d pose result
    comotion_path = data_dir / 'comotion' / 'comotion_result.pkl'
    assert comotion_path.exists()
    with open(comotion_path, 'rb') as file:
        comotion_result = pickle.load(file)

    # load 2d pose result
    dwpose_path =  data_dir / 'dwpose' / 'dwpose_result.pkl'
    assert dwpose_path.exists()
    with open(dwpose_path, 'rb') as file:
        dwpose_result = pickle.load(file)
    
    # compare comotion and dwpose to select pose candidate
    for shot_id, shot_co_result in comotion_result.items():
        if shot_id not in dwpose_result: # no detection in shot
            continue
        shot_dw_result = dwpose_result[shot_id]
        
        for pnum, person_co_result in shot_co_result.items():
            if pnum not in shot_dw_result: # no detection in person
                continue
            person_dw_result = shot_dw_result[pnum]
            
            for fname, frame_co_result in person_co_result.items():
                if fname not in person_dw_result: # no person detected in frame
                    continue
                frame_dw_dict = person_dw_result[fname]
                dw_j2d_cands = frame_dw_dict['body'] # dwpose candidates
                
                assert len(dw_j2d_cands.shape) == 3
                num_cands = dw_j2d_cands.shape[0]

                if num_cands == 0: # no person detected in frame
                    continue
                elif num_cands == 1: # one person detected in frame
                    dw_j2d = dw_j2d_cands[0] # (24,3)
                elif num_cands > 1: # multiple persons detected in frame
                    # find best matching candidate
                    co_j2d = frame_co_result['j2d'] # comotion 2d pose
                    co_j2d = torch.stack(smpl_to_op25(co_j2d), dim=0)
                    
                    min_distance = float('inf')
                    min_id = -1

                    for j2d_id, dw_j2d_cand in enumerate(dw_j2d_cands):

                        dw_j2d_cand = torch.stack(dwpose_to_op25(dw_j2d_cand), dim=0)

                        distance = 0
                        j_cnt = 0

                        dw_j2d_cand = dwpose_to_op25(dw_j2d_cand)
                        for co_jnt, dw_jnt in zip(co_j2d, dw_j2d_cand): 
                            if (~dw_jnt.isnan().all() and ~co_jnt.isnan().all()):
                                distance += torch.sqrt(((co_jnt[:2] - dw_jnt[:2])**2).sum()) # calculate L2 distance between valid joints
                                j_cnt += 1
                        if j_cnt > 0:                                  
                            distance /= j_cnt # average distance
                            if distance < min_distance:
                                min_distance = distance
                                min_id = j2d_id
                        
                    if min_id < 0: # No valid body pose detection exists
                        continue
                    else:
                        dw_j2d = dw_j2d_cands[min_id] # select best match

                else:
                    raise ValueError(f"Invalid number of candidates: {num_cands}")
                
                # save as openpose 25 format
                op25_dw_j2d = torch.stack(dwpose_to_op25(dw_j2d), dim=0)
                op25_co_j2d = torch.stack(smpl_to_op25(frame_co_result['j2d']), dim=0)
                
                # merge
                comotion_result[shot_id][pnum][fname]['j2d'] = op25_co_j2d
                comotion_result[shot_id][pnum][fname]['body'] = op25_dw_j2d
                comotion_result[shot_id][pnum][fname]['left_hand'] = frame_dw_dict['left_hand']
                comotion_result[shot_id][pnum][fname]['right_hand'] = frame_dw_dict['right_hand']
                comotion_result[shot_id][pnum][fname]['face'] = frame_dw_dict['face']

    # save merged results
    merged_result = comotion_result.copy()
    output_path = data_dir / 'merged_result.pkl'
    with open(output_path, 'wb') as file:
        pickle.dump(merged_result, file)