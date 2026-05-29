import os
import cv2
import numpy as np
import tyro
from tqdm import tqdm

def main(
    input_paths: list[str],
    merge_path: str,
    output_path: str
):
    if len(input_paths) == 1: # revert only one mask
        for filename in tqdm(os.listdir(input_paths[0]), total=len(input_paths[0]), desc='revert mask ...'):
            if filename.endswith('png') or filename.endswith('jpg'):
                mask_file_path = os.path.join(input_paths[0], filename)
                mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(os.path.join(merge_path, filename), mask)
                inverted_mask = 255 - mask
                save_file_path = os.path.join(output_path, filename+'.png') # colmap format = *.png.png
                cv2.imwrite(save_file_path, inverted_mask)
    elif len(input_paths) == 2: # merge two masks and revert
        input_list0 = [f for f in os.listdir(input_paths[0]) if f.endswith(".png") or f.endswith(".jpg")]
        input_list1 = [f for f in os.listdir(input_paths[1]) if f.endswith(".png") or f.endswith(".jpg")]
        assert len(input_list0) == len(input_list1)
        for filename0, filename1 in tqdm(zip(input_list0, input_list1), total=len(input_list0), desc='revert mask'):
            mask_file_path0 = os.path.join(input_paths[0], filename0)
            mask_file_path1 = os.path.join(input_paths[1], filename1)
            mask0 = cv2.imread(mask_file_path0, cv2.IMREAD_GRAYSCALE)
            mask1 = cv2.imread(mask_file_path1, cv2.IMREAD_GRAYSCALE)
            if mask0.shape != mask1.shape: # resize to equal size
                if mask0.shape[0] < mask1.shape[0] or mask0.shape[1] < mask1.shape[1]:
                    mask0 = cv2.resize(mask0, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_LINEAR)
                else:
                    mask1 = cv2.resize(mask1, (mask0.shape[1], mask0.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask0[mask0 != 0] = 1
            mask1[mask1 != 0] = 1
            merge_mask = (np.bitwise_or(mask0, mask1)*255).astype(np.uint8)
            cv2.imwrite(os.path.join(merge_path, filename0), merge_mask)
            inverted_mask = 255 - merge_mask
            save_file_path = os.path.join(output_path, filename0+'.png') # colmap format = *.png.png
            cv2.imwrite(save_file_path, inverted_mask)
    else:
        raise Exception('Support only one or two mask path.')

if __name__=="__main__":
    tyro.cli(main)