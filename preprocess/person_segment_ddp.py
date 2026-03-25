import torch
import argparse
import cv2
import pickle
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from segment_anything import SamPredictor, build_sam
from tqdm import tqdm

class ModelWrapper:
    def __init__(self, ckpt_path, gpu_id):
        self.device = f'cuda:{gpu_id}'
        self.model = SamPredictor(build_sam(checkpoint=ckpt_path).to(self.device))
    
    def __call__(self, img, np_jnts):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.model.set_image(img)    
        masks, _, _ = self.model.predict(point_coords=np_jnts, point_labels=np.ones(np_jnts.shape[0]))
        mask = masks.sum(axis=0) > 0
        return (mask * 255).astype(np.uint8)

def process_batch(img_list, overall_mask_list, track_results, sam_ckpt, output_dir, gpu_num, gpu_id):
    pipe = ModelWrapper(sam_ckpt, gpu_id)
    
    img_dict = dict()
    overall_mask_dict = dict()
    print(f"Loading images in gpu {gpu_num}.")
    for file_path in img_list:
        img = cv2.imread(file_path)
        fname = str(file_path.name.split('.')[0])
        img_dict[fname] = img
    
    print(f"Loading masks in gpu {gpu_num}.")
    for file_path in overall_mask_list:
        overall_mask = cv2.imread(file_path)
        overall_mask = (overall_mask / 255.).astype(np.uint8)
        overall_mask = overall_mask[:, :, 0]
        fname = str(file_path.name.split('.')[0])
        overall_mask_dict[fname] = overall_mask
    
    print(f"Start segmentation in gpu {gpu_num}.")
    for shot_id, shot_result in track_results.items(): # 'bbox', 'smpl_param', 'j3d', 'j2d', 'body', 'left_hand', 'right_hand', 'face'
        for pnum, track_result in shot_result.items():
            for fname, frame_result in tqdm(track_result.items(), desc=f"GPU {gpu_num} - person {pnum}"):
                if fname not in img_dict.keys(): # skip if not in batch
                    continue
                x1 = frame_result['bbox'][0] # bbox corners
                y1 = frame_result['bbox'][1]
                x2 = x1 + frame_result['bbox'][2]
                y2 = y1 + frame_result['bbox'][3]

                if 'body' not in frame_result.keys(): # skip if 2D keypoints not detected
                    continue
                
                op_j2ds = frame_result['body']

                input_points = []
                for kpt in op_j2ds: # select joints within bbox
                    if kpt.isnan().all():
                        continue
                    if (x1 <= kpt[0] and kpt[0] <= x2) and (y1 <= kpt[1] and kpt[1] <= y2 ):
                        input_points.append(kpt[:2].numpy())
                
                input_img = img_dict[fname]
                input_overall_mask = overall_mask_dict[fname]
                
                np_input_points = np.array(input_points)
                if np_input_points.shape[0] != 0:
                    mask = pipe(input_img, np_input_points)
                else:
                    mask = np.zeros(input_img.shape[:2])
                mask = mask * input_overall_mask
                save_path = output_dir / f'{shot_id:03d}_{pnum:03d}' / f'{fname}.png'
                cv2.imwrite(save_path, mask)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="generate masks per actor")
    parser.add_argument('--data_dir', type=str, help='path to data directory')
    parser.add_argument('--sam_checkpoint_path', type=str, help="sam checkpoint dir")
    parser.add_argument('--gpus', type=str)
    args = parser.parse_args()

    gpu_nums = args.gpus.split(',')
    gpu_size = len(gpu_nums)
    gpu_ids = [gid for gid in range(gpu_size)]
    ctx = mp.get_context('spawn')

    img_dir = Path(args.data_dir) / 'frames'
    img_list = sorted(list(img_dir.glob('*.png'))+list(img_dir.glob('*.jpg')))

    mask_dir = Path(args.data_dir) / 'actor_masks'
    mask_list = sorted(list(mask_dir.glob('*.png'))+list(mask_dir.glob('*.jpg')))
    
    # load merged results
    merged_path = Path(args.data_dir) / 'merged_result.pkl'
    assert merged_path.exists()
    
    print("Loading merged data ...")
    with open(merged_path, 'rb') as file:
        merged_result = pickle.load(file)

    # make output directory
    output_dir = Path(args.data_dir) / 'personal_masks'
    output_dir.mkdir(exist_ok=True)
    for shot_id, shot_result in merged_result.items():
        for pnum in shot_result.keys():
            peronal_dir = output_dir / f'{shot_id:03d}_{pnum:03d}'
            peronal_dir.mkdir(exist_ok=True)

    batch_size = len(img_list) // gpu_size + (1 if len(img_list) % gpu_size != 0 else 0)
    
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
            process_arg = (img_batch, 
                            mask_batch,
                            merged_result, 
                            args.sam_checkpoint_path, 
                            output_dir, 
                            gpu_nums[gpu_id],
                            gpu_id)
            process_list.append(executor.submit(process_batch, *process_arg))
        
        for process in process_list:
            process.result()