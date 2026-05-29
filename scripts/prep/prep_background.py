import os
import subprocess
import pathlib
import tyro
from config.prep import PrepConfig

def run_command(cmd):
    try:
        process = subprocess.Popen(cmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, preexec_fn=os.setsid)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            print(f"[ERROR] Command failed with return code {process.returncode}")
            exit(1)
    except Exception as e:
        print(f"[ERROR] Command failed with error: {str(e)}")
        exit(1)

def main(cfg: PrepConfig):
    data = cfg.data
    gpus = cfg.gpus

    data_path = pathlib.Path(os.path.join("demo",data)) # input path
    bg_path = data_path / 'background'
    assert bg_path.exists()
    img_path = bg_path / 'images'
    assert img_path.exists()
    for img_file in img_path.glob('*'):
        if img_file.suffix.lower() != '.png':
            raise ValueError(f"{img_file} is not a PNG file. All images must be in PNG format.")
    
    fg_mask_path = bg_path / 'foreground_masks'
    sam_path = pathlib.Path("submodules/Grounded-Segment-Anything")

    actor_mask_path = bg_path / 'actor_masks'
    actor_mask_path.mkdir(exist_ok=True)
    
    merge_mask_path = bg_path / 'merged_masks'
    merge_mask_path.mkdir(exist_ok=True)
    
    colmap_mask_path = bg_path / 'colmap_masks'
    colmap_mask_path.mkdir(exist_ok=True)

    inpaint_img_path = bg_path / 'inpainted_images'
    inpaint_img_path.mkdir(exist_ok=True)
    
    # ------------------------ 1. generate stage masks ------------------------
    print("[INFO] Start generating stage masks.")
    cmd = (
        f"conda run --no-capture-output -n grounded-sam "
        f"CUDA_VISIBLE_DEVICES={gpus} python -u preprocess/grounded_sam_ddp.py "
        f"--config {sam_path}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py "
        f"--grounded_checkpoint {sam_path}/groundingdino_swint_ogc.pth "
        f"--sam_checkpoint {sam_path}/sam_vit_h_4b8939.pth "
        f"--input_dir {img_path} "
        f"--output_dir {actor_mask_path} "
        f"--text_prompt human "
        f"--box_threshold {cfg.box_threshold} "
        f"--gpus {gpus} "
    )
    run_command(cmd)
    print("[INFO] Finished generating stage masks.")
    
    # ------------------------ 2. generate masks for colmap ------------------------
    if not fg_mask_path.exists():
        print("[INFO] Start generating colmap masks.")
        cmd = (
            f"conda run --no-capture-output -n sm3r "
            f"CUDA_VISIBLE_DEVICES={gpus} python -u preprocess/colmap_masks.py "
            f"--input_paths {actor_mask_path} "
            f"--merge_path {merge_mask_path} "
            f"--output_path {colmap_mask_path} "
        )
    else:
        print("[INFO] Start generating colmap masks with foreground.")
        cmd = (
            f"conda run --no-capture-output -n sm3r "
            f"CUDA_VISIBLE_DEVICES={gpus} python -u preprocess/colmap_masks.py "
            f"--input_paths {actor_mask_path} {fg_mask_path} "
            f"--merge_path {merge_mask_path} "
            f"--output_path {colmap_mask_path} "
        )
    run_command(cmd)
    print("[INFO] Finished generating colmap masks.")
    
    # ------------------------ 3. inpaint stage foreground ------------------------
    print("[INFO] Start inpainting stage foreground.")
    cmd = (
        f"conda run --no-capture-output -n sm3r "
        f"CUDA_VISIBLE_DEVICES={gpus} python -u preprocess/objectclear_ddp.py "
        f"--input_path {img_path} "
        f"--mask_path {merge_mask_path} "
        f"--output_path {inpaint_img_path} "
        f"--gpus {gpus} "
    )
    run_command(cmd)
    print("[INFO] Finished inpainting stage foreground.")

if __name__ == '__main__':
    main(tyro.cli(PrepConfig))    
    