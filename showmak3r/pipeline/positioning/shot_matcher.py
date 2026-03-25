import numpy as np

def associate_actors(cfg, shot_dicts, frame_boundaries):
    '''
    Associate actors across the shots.
    
    Return:
        - shot_dict: associated shots dict
        - mapped_dict: dict of mapped pairs (old_key: new_key)
    '''
    # initialize with the first shot
    people_dict = shot_dicts[0].copy()
    mapped_dict = dict()
    # merge consecutive shots
    for shot_id in range(len(shot_dicts)-1):
        people_dict, mapped_pairs = merge_two_shots(people_dict, shot_dicts[shot_id+1], 
                                      frame_boundaries[shot_id+1], frame_boundaries[shot_id+2], 
                                      cfg.match_threshold)
        # update mapped_dict
        for old_key, new_key in mapped_pairs.items():
            if old_key not in mapped_dict.keys():
                mapped_dict[old_key] = new_key

    # return as shot dict with only one shot
    return {0: people_dict}, mapped_dict

def merge_two_shots(current_shot, next_shot, cur_boundary:int, next_boundary:int, match_threshold:float):
    '''
    1. Compare every possible matches
    2. Choose the shortest distance & Below threshold? -> identify as same person
    3. Interpolate rest of the humans across the frames
    '''
    shot_before = current_shot.copy()
    shot_after = next_shot.copy()

    cur_fname_before = f'frame_{cur_boundary:04}'
    cur_fname_after = f'frame_{cur_boundary+1:04}'
    
    # SMPL position dicts at shot boundary
    smpls_before_dict = dict()
    smpls_after_dict = dict()
    # get SMPL translation at last frame of current shot
    for pnum, person_dict in shot_before.items():    
        smpls_before_dict[pnum] = person_dict[cur_fname_before]['smpl_param'][1:4]
    # get SMPL translation at first frame of next shot
    for pnum, person_dict in shot_after.items():    
        smpls_after_dict[pnum] = person_dict[cur_fname_after]['smpl_param'][1:4]

    match_dict = {} # dict of matched person pairs = {pnum_before: pnum_after}
    # person number list of before and after shot
    pnums_before_keys = list(smpls_before_dict.keys())
    pnums_after_keys = list(smpls_after_dict.keys())
    
    # calculate distance between all possible matches
    for pnum_before in pnums_before_keys:
        min_distance = float('inf')
        match_pnum_candidate = -1
        for pnum_after in pnums_after_keys:
            # calculate L2 distance between SMPL translation at shot boundary
            l2_distance = np.linalg.norm(smpls_before_dict[pnum_before] - smpls_after_dict[pnum_after])
            print(f"pair {pnum_before} and {pnum_after} distance: {l2_distance}")
            # pick the closest match
            if l2_distance < min_distance:
                match_pnum_candidate = pnum_after
                min_distance = l2_distance
        # if the distance is larger than threshold, identify as different person
        if min_distance<match_threshold:
            print(f"matching {pnum_before} and {match_pnum_candidate}")
            pnums_after_keys.remove(match_pnum_candidate)
            # add to the matched list
            match_dict[pnum_before] = match_pnum_candidate

    # -------------------- extrapolate to the shot after --------------------
    for pnum_before in list(smpls_before_dict.keys()): 
        if pnum_before in match_dict.keys(): # skip matched actors
            continue
        person_dict = shot_before[pnum_before]
        # get smpl param at last frame of the shot
        save_dict = {
            'bbox': None,
            'smpl_param': person_dict[cur_fname_before]['smpl_param'],
            'j2d': None,
            'body': None,
        }
        # fill in next shot frames
        for fid in range(cur_boundary, next_boundary): # cur_boundary is in frame name format = fid + 1
            fname = f'frame_{fid+1:04}' 
            # extend current person_dict to next shot
            shot_before[pnum_before][fname] = save_dict

    # -------------------- extrapolate to the shot before --------------------
    for pnum_after in list(smpls_after_dict.keys()): 
        if pnum_after in match_dict.values(): # skip matched actors
            continue
        person_dict = shot_after[pnum_after]
        # get smpl param at first frame of the shot
        save_dict = {
            'bbox': None,
            'smpl_param': person_dict[cur_fname_after]['smpl_param'],
            'j2d': None,
            'body': None,
        }
        # fill in next shot frames
        for fid in range(0, cur_boundary): # all shots before have been merged
            fname = f'frame_{fid+1:04}' 
            # extend current person_dict to prev shot
            shot_after[pnum_after][fname] = save_dict
    
    # -------------------- merge matched actors --------------------
    # merge matched actors
    for pnum_before, pnum_after in match_dict.items():
        shot_before[pnum_before].update(shot_after[pnum_after])
        shot_after.pop(pnum_after)
    
    # add extrapolated actors
    for pnum in shot_after.keys():
        shot_before[pnum] = shot_after[pnum]

    # sort person ids
    merged_shot_dict = {}
    mapped_pairs = {}
    for new_key, old_key in enumerate(sorted(shot_before.keys()), start=1): # update pnum
        merged_shot_dict[new_key] = shot_before[old_key]
        mapped_pairs[old_key] = new_key
        # if there is a match, add the corresponding pair as well
        if old_key in match_dict.keys():
            mapped_pairs[match_dict[old_key]] = new_key

    return merged_shot_dict, mapped_pairs
