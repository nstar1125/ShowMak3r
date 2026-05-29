import argparse
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import torch
import multiprocessing as mp

from preprocess.utils.image_utils import get_crop_img
from controlnet_aux import DWposeDetector
from controlnet_aux.util import HWC3, resize_image
from controlnet_aux.dwpose import util as dw_util

class ModelWrapper:
    def __init__(self, model,
        body_threshold=0.3,
        conf_threshold=0.5,
        boundary_weight=1.0,
        boundary_length=0.1,
    ):
        self.model = model
        self.body_threshold = body_threshold
        self.conf_threshold = conf_threshold
        self.boundary_weight = boundary_weight
        self.boundary_length = boundary_length
    
    def __call__(self, 
        img, 
        mask,
        bbox=None, 
        detect_resolution=512, 
        image_resolution=512, 
        output_type="pil", 
        **kwargs):
        input_img = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2BGR) # convert to BGR for DWPose format
        input_mask = np.array(mask, dtype=np.float32) / 255.0 # normalize mask
        input_mask = input_mask.astype(np.uint8)

        input_img = HWC3(input_img)
        raw_input_img = input_img
        H, W, C = input_img.shape
        raw_H = H
        raw_W = W
        
        # 1. crop image and mask to bbox and resize to detect_resolution
        if bbox is not None:
            input_img, new_bbox = get_crop_img(input_img, bbox, rescale=1.1, resize=-1, get_new_bbox=True)
            W = new_bbox[-2]
            H = new_bbox[-1]
        input_img = resize_image(input_img, detect_resolution)

        with torch.no_grad():
            # 2. detect pose
            candidate, confidence = self.model.pose_estimation(input_img) # (1, 134, 2) / (1, 134)
            nums, keys, locs = candidate.shape # (1, 134, 2)
            candidate[..., 0] /= float(detect_resolution) # normalize x
            candidate[..., 1] /= float(detect_resolution) # normalize y
            body = candidate[:,:18].copy() # (1, 18, 2)
            body = body.reshape(18*nums, locs) # (18, 2)
            body_score = np.copy(confidence[:,:18]) # (1, 18)

            # 3. filter with confidence
            for i in range(len(body_score)): # per candidate
                for j in range(len(body_score[i])): # per joints
                    # lower confidence score if near image boundaries
                    x = candidate[i][j][0]
                    y = candidate[i][j][1]
                    if x < self.boundary_length or x > 1-self.boundary_length or y < self.boundary_length or y > 1-self.boundary_length:
                        body_score[i][j] *= self.boundary_weight
                        confidence[i][j] *= self.boundary_weight
                    
                    # confidence filtering
                    if body_score[i][j] > self.body_threshold: 
                        body_score[i][j] = int(18*i+j)
                    else:
                        body_score[i][j] = -1
            
            un_visible = confidence < self.conf_threshold
            candidate[un_visible] = -1 # confidence masking

            foot = candidate[:,18:24] # (1, 6, 2)
            faces = candidate[:,24:92] # (1, 68, 2)
            hands = candidate[:,92:113] # left?
            hands = np.vstack([hands, candidate[:,113:]]) # right? -> (1*2, 68, 2)
            bodies = dict(candidate=body, score=body_score) # -1 = invisible
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            # 4. pose to detection map
            detected_map = self.draw_pose(pose, H, W)
            detected_map = HWC3(detected_map)

            # 5. resize detection map to bbox
            if bbox is not None:
                black_bg = np.zeros(raw_input_img.shape, dtype=np.uint8)
                x,y,w,h = new_bbox

                if x+w>raw_W:
                    w = raw_W-x-1
                    detected_map = detected_map[:, :w]
                if y+h>raw_H:
                    h = raw_H-y-1
                    detected_map = detected_map[:h]
                if x<0:
                    detected_map = detected_map[:, -x:]
                    w = w + x
                    x = 0
                if y<0:
                    detected_map = detected_map[-y:]
                    h = h + y
                    y = 0
                black_bg[y:y+h, x:x+w] = detected_map
                detected_map = black_bg
            else:
                detected_map = cv2.resize(detected_map, (W,H), interpolation=cv2.INTER_LINEAR)
                w=raw_W
                h=raw_H
            
            # 6. draw open pose image on top
            op_img_mask = (detected_map.sum(-1) > 0)
            detected_map = raw_input_img[...,::-1] * (1-op_img_mask[..., None]) + detected_map * op_img_mask[..., None]
            detected_map = detected_map.astype(np.uint8)
            if output_type=='pil':
                detected_map = Image.fromarray(detected_map)

            # 7. resize pose to bbox
            body_list = []
            left_hand_list = []
            right_hand_list = []
            face_list = []
            for cand, conf in zip(candidate, confidence): # for people in images
                bodies = cand[:24] # 24
                bodies_score = conf[:24] 
                faces = cand[24:92] # 68
                faces_score = conf[24:92] 
                l_hands = cand[92:113]
                l_hands_score = conf[92:113] 
                r_hands = cand[113:]
                r_hands_score = conf[113:]
                
                is_bbox = bbox is not None
                
                # resize to bbox
                body = self.resize_to_bbox(bodies, bodies_score, new_bbox, is_bbox)
                left_hand = self.resize_to_bbox(l_hands, l_hands_score, new_bbox, is_bbox)
                right_hand = self.resize_to_bbox(r_hands, r_hands_score, new_bbox, is_bbox)
                face = self.resize_to_bbox(faces, faces_score, new_bbox, is_bbox)

                # filter our joints out side the box                
                body = self.filter_out_mask(body, input_mask)
                left_hand = self.filter_out_mask(left_hand, input_mask)
                right_hand = self.filter_out_mask(right_hand, input_mask)
                face = self.filter_out_mask(face, input_mask)
                
                # replace None with nan
                body = [[np.nan, np.nan, np.nan] if joint is None else joint for joint in body]
                left_hand = [[np.nan, np.nan, np.nan] if joint is None else joint for joint in left_hand]
                right_hand = [[np.nan, np.nan, np.nan] if joint is None else joint for joint in right_hand]
                face = [[np.nan, np.nan, np.nan] if joint is None else joint for joint in face]
                
                body_list.append(torch.tensor(body, dtype=torch.float32))
                left_hand_list.append(torch.tensor(left_hand, dtype=torch.float32))
                right_hand_list.append(torch.tensor(right_hand, dtype=torch.float32))
                face_list.append(torch.tensor(face, dtype=torch.float32))
            
            body_tensor = torch.stack(body_list)
            left_hand_tensor = torch.stack(left_hand_list)
            right_hand_tensor = torch.stack(right_hand_list)
            face_tensor = torch.stack(face_list)

            new_poses = dict(
                body = body_tensor,
                left_hand = left_hand_tensor,
                right_hand = right_hand_tensor,
                face = face_tensor
            )
            
            return detected_map, new_poses

    def draw_pose(self, pose, H, W): # return detection map
        bodies = pose['bodies']
        hands = pose['hands']
        faces = pose['faces']
        candidates = bodies['candidate']
        score = bodies['score']

        canvas = np.zeros(shape=(H,W,3), dtype=np.uint8)
        canvas = dw_util.draw_bodypose(canvas, candidates, score)
        canvas = dw_util.draw_handpose(canvas, hands)
        canvas = dw_util.draw_facepose(canvas, faces)
        
        return canvas
    
    def resize_to_bbox(self, pose_set, score_set, new_bbox, is_bbox):
        new_pose_set = []
        for kpt, s in zip(pose_set, score_set):
            if kpt[0] < 0 and kpt[1] < 0:
                new_pose_set.append(None)
                continue
            save_joint = [kpt[0], kpt[1], s]
            if is_bbox:
                x1, y1, w, h = new_bbox
                save_joint[0] = save_joint[0] * w + x1
                save_joint[1] = save_joint[1] * h + y1
            new_pose_set.append(save_joint)
        return new_pose_set
    
    def filter_out_mask(self, joint_list, mask):
        filtered_joint_list = []
        H, W = mask.shape[:2]
        for joint in joint_list:
            if joint is None:
                filtered_joint_list.append(None)
                continue
            # Clip coordinates to mask boundaries
            x = min(max(int(joint[0]), 0), W-1)
            y = min(max(int(joint[1]), 0), H-1)
            if mask[y, x].all(): # is in actor region
                filtered_joint_list.append(joint)
            else:
                filtered_joint_list.append(None)
        return filtered_joint_list
    
def process_batch(args, img_path_batch, mask_path_batch, comotion_result, visualize, out_dir, gpu_id, gpu_num):
    device = f"cuda:{gpu_id}"
    print(f"\nProcessing batch on GPU {gpu_num}. batch size: {len(img_path_batch)}.")

    # load model
    pose_estimator = DWposeDetector(
        det_ckpt="https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
        pose_ckpt="https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth",
        device=device
    )
    pipe = ModelWrapper(
        pose_estimator,
        body_threshold=args.body_threshold,
        conf_threshold=args.conf_threshold,
        boundary_weight=args.boundary_weight,
        boundary_length=args.boundary_length,
    )

    # extract bbox
    track_bbox_dict = dict() # bbox result
    dwpose_est_dict = dict() # dwpose result
    for shot_id, shot_result in comotion_result.items(): # per shot
        track_bbox_dict[shot_id] = dict()
        dwpose_est_dict[shot_id] = dict()
        for pnum, person_result in shot_result.items(): # per person
            person_bbox_dict = dict()
            for fname, frame_result in person_result.items():
                person_bbox_dict[fname] = frame_result['bbox']
            track_bbox_dict[shot_id][pnum] = person_bbox_dict
            dwpose_est_dict[shot_id][pnum] = dict()        
        
    # process batch
    for img_path, mask_path in tqdm(zip(img_path_batch, mask_path_batch), desc=f'GPU {gpu_num}', total=len(img_path_batch)):
        fname = str(img_path.name.split(".")[0]) # ex) frame_0001
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        img_resolution = min(img.size)
        for shot_id, shot_bbox in track_bbox_dict.items():
            for pnum, person_bbox_dict in shot_bbox.items():
                if fname not in person_bbox_dict: # skip frames not in batch
                    continue
                # DWPose estimation
                bbox = person_bbox_dict[fname]
                img_w_pose, poses = pipe(
                    img,
                    mask,
                    bbox,
                    detect_resolution=512,
                    image_resolution=img_resolution,
                )
                dwpose_est_dict[shot_id][pnum][fname] = poses # body, left_hand, right_hand, face
                if visualize:
                    img_w_pose.save(out_dir / f'{shot_id:03}_{pnum:03}' / img_path.name)
    
    # save dwpose results
    return dwpose_est_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--visualize", action='store_true', help="visualize the pose estimation results.")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--body_threshold", type=float, default=0.3)
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--boundary_weight", type=float, default=1.0)
    parser.add_argument("--boundary_length", type=float, default=0.1)
    args = parser.parse_args()

    gpu_nums = args.gpus.split(',')
    gpu_size = len(gpu_nums)
    gpu_ids = [gid for gid in range(gpu_size)]

    ctx = mp.get_context('spawn')

    # load images
    img_dir = Path(args.data_dir) / 'frames'
    img_list = sorted(list(img_dir.glob("*.png")))

    # load mask
    mask_dir = Path(args.data_dir) / 'actor_masks'
    mask_list = sorted(list(mask_dir.glob("*.png")))

    # load comotion result
    comotion_path = Path(args.data_dir) / "comotion" / "comotion_result.pkl"
    with open(comotion_path, 'rb') as f:
        comotion_result = pickle.load(f)
    
    # make output dir
    out_dir = Path(args.data_dir) / 'dwpose'
    for shot_id in comotion_result.keys():
        for pnum in comotion_result[shot_id].keys(): # person num
            person_out_dir = out_dir / f'{shot_id:03}_{pnum:03}'
            person_out_dir.mkdir(exist_ok=True, parents=True)
    
    batch_size = len(img_list) // gpu_size + (1 if len(img_list) % gpu_size != 0 else 0)
    
    process_outputs = []
    with ProcessPoolExecutor(max_workers=gpu_size, mp_context=ctx) as executor:
        process_list = []
        for gpu_id in gpu_ids:
            gpu_id = int(gpu_id) % gpu_size
            if gpu_id == gpu_size-1: # last batch
                img_batch = img_list[gpu_id*batch_size:]
                mask_batch = mask_list[gpu_id*batch_size:]
            else:
                img_batch = img_list[gpu_id*batch_size:(gpu_id+1)*batch_size]
                mask_batch = mask_list[gpu_id*batch_size:(gpu_id+1)*batch_size]
            process_arg = (
                args,
                img_batch,
                mask_batch,
                comotion_result,
                args.visualize,
                out_dir,
                gpu_id,
                gpu_nums[gpu_id],
            )
            process_list.append(executor.submit(process_batch, *process_arg))
        
        for process in process_list:
            process_output = process.result()
            process_outputs.append(process_output)
    
    # initialize dwpose dict
    dwpose_dict = dict()
    for shot_id, shot_result in comotion_result.items():
        dwpose_dict[shot_id] = dict()
        for pnum in shot_result.keys():
            dwpose_dict[shot_id][pnum] = dict()
    
    # merge each process outputs
    for dwpose_est_dict in process_outputs: # per gpu
        for shot_id, shot_result in dwpose_est_dict.items(): # per shot
            for pnum, person_result in shot_result.items(): # per person
                for frame_name, poses in person_result.items(): # per frame
                    dwpose_dict[shot_id][pnum][frame_name] = poses # body, left_hand, right_hand, face
                    
    # save dwpose result
    output_path = Path(args.data_dir) / "dwpose" / f"dwpose_result.pkl"
    with open(output_path, 'wb') as file:
        pickle.dump(dwpose_dict, file)