import os
import argparse
import glob
import torch
from PIL import Image
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.getcwd(), "submodules/ObjectClear"))
from objectclear.pipelines import ObjectClearPipeline
from objectclear.utils import resize_by_short_side

def process_batch(args, gpu_id, gpu_num, img_batch, mask_batch):
    device = torch.device(f'cuda:{gpu_id}')
    print(f"\nProcessing batch on GPU {gpu_num}. batch size: {len(img_batch)}.")
    # ------------------ set up ObjectClear pipeline -------------------
    torch_dtype = torch.float16 if args.use_fp16 else torch.float32
    variant = "fp16" if args.use_fp16 else None
    generator = torch.Generator(device=device).manual_seed(args.seed)
    pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
        "jixin0101/ObjectClear",
        torch_dtype=torch_dtype,
        apply_attention_guided_fusion=True,
        cache_dir=args.cache_dir,
        variant=variant,
    )
    pipe.to(device)

    # ------------------------ start batch processing ------------------------
    for i, (img_path, mask_path) in tqdm(enumerate(zip(img_batch, mask_batch)), desc=f"GPU {gpu_id}", total=len(img_batch)):
        img_name = os.path.basename(img_path)
        basename, ext = os.path.splitext(img_name)
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image_or = image.copy()
        
        # Our model was trained on 512×512 resolution.
        # Resizing the input so that the **shorter side is 512** helps achieve the best performance.
        image = resize_by_short_side(image, 512, resample=Image.BICUBIC)
        mask = resize_by_short_side(mask, 512, resample=Image.NEAREST)
        
        w, h = image.size
    
        result = pipe(
            prompt="remove the instance of object",
            image=image,
            mask_image=mask,
            generator=generator,
            num_inference_steps=args.steps,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            height=h,
            width=w,
            return_attn_map=False,
        )
        
        fused_img_pil = result.images[0]

        # save results
        save_path = os.path.join(args.output_path, f'{basename}.png')
        fused_img_pil = fused_img_pil.resize(image_or.size)
        fused_img_pil.save(save_path)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()

    # directory
    parser.add_argument('-i', '--input_path', type=str, required=True, 
                        help='Input folder.')
    parser.add_argument('-m', '--mask_path', type=str, required=True,
                        help='Input mask folder.')
    parser.add_argument('-o', '--output_path', type=str, required=True, 
                        help='Output folder.')
    # parameters
    parser.add_argument('--cache_dir', type=str, default=None,
                        help="Path to cache directory")
    parser.add_argument('--use_fp16', action='store_true', 
                        help='Use float16 for inference')
    parser.add_argument('--seed', type=int, default=42, 
                    help='Random seed for torch.Generator. Default: 42')
    parser.add_argument('--steps', type=int, default=20, 
                        help='Number of diffusion inference steps. Default: 20')
    parser.add_argument('--strength', type=float, default=0.99, 
                        help='Strength of the denoising process. Default: 0.99')
    parser.add_argument('--guidance_scale', type=float, default=2.5, 
                        help='CFG guidance scale. Default: 2.5')
    parser.add_argument("--gpus", type=str, required=True, help="gpu ids")
    args = parser.parse_args()
    
    gpu_nums = args.gpus.split(',')
    gpu_size = len(gpu_nums)
    gpu_ids = [gid for gid in range(gpu_size)]
    ctx = mp.get_context('spawn')

    # ------------------------ input & output ------------------------
    # make dir
    os.makedirs(args.output_path, exist_ok=True)
    
    # fetch images
    if not os.path.isdir(args.input_path):
        img_targets = [args.input_path]
    else:
        img_targets = [
            f for f in sorted(os.listdir(args.input_path)) if not os.path.isdir(os.path.join(args.input_path, f))
        ]
        img_targets = [os.path.join(args.input_path, f) for f in img_targets]
    
    # fetch masks
    if not os.path.isdir(args.mask_path):
        mask_targets = [args.mask_path]
    else:
        mask_targets = [
            f for f in sorted(os.listdir(args.mask_path)) if not os.path.isdir(os.path.join(args.mask_path, f))
        ]
        mask_targets = [os.path.join(args.mask_path, f) for f in mask_targets]
    
    if len(img_targets) != len(mask_targets):
        raise ValueError(f"Mismatch between input images ({len(img_targets)}) and masks ({len(mask_targets)}).")
    
    batch_size = len(img_targets) // gpu_size + (1 if len(img_targets) % gpu_size != 0 else 0)

    # ------------------------ distribute tasks ------------------------
    with ProcessPoolExecutor(max_workers=gpu_size, mp_context=ctx) as executor:
        process_list = []
        for gpu_id in gpu_ids:
            if gpu_id == gpu_size - 1: # last batch
                img_batch = img_targets[gpu_id * batch_size:]   
                mask_batch = mask_targets[gpu_id * batch_size:]
            else: # other batches
                img_batch = img_targets[gpu_id * batch_size : (gpu_id + 1) * batch_size]
                mask_batch = mask_targets[gpu_id * batch_size : (gpu_id + 1) * batch_size]
            process_arg = (
                args,
                gpu_id,
                gpu_nums[gpu_id],
                img_batch,
                mask_batch
            )
            process_list.append(executor.submit(process_batch, *process_arg))
        for process in process_list:
            process.result() # wait for all process to complete