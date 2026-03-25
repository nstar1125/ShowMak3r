import tyro
import pathlib
import os
import subprocess
import time
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
    
    # load paths
    data_path = pathlib.Path(os.path.join("demo",data)) # input path
    composite_path = data_path / 'composite'
    composite_path.mkdir(exist_ok=True)
    
    video_path = data_path / 'video'
    bg_path = data_path / 'background'

    # make directories
    composite_img_path = composite_path / 'images'
    composite_img_path.mkdir(exist_ok=True)
    composite_mask_path = composite_path / 'actor_masks'
    composite_mask_path.mkdir(exist_ok=True)
    composite_foreground_mask_path = composite_path / 'foreground_masks'
    composite_foreground_mask_path.mkdir(exist_ok=True)
    composite_colmap_path = composite_path / 'colmap_masks'
    composite_colmap_path.mkdir(exist_ok=True)
    composite_inpaint_path = composite_path / 'inpainted_images'
    composite_inpaint_path.mkdir(exist_ok=True)
    (composite_path / 'undistorted' / 'sparse' / '0').mkdir(parents=True, exist_ok=True)

    # ------------------------ 1. copy video files to composite ------------------------
    for file in video_path.glob('frames/*.png'):
        os.system(f'cp {file} {composite_img_path}')
    print("[INFO] Copied video frames to composite")
    
    for file in video_path.glob('actor_masks/*.png'):
        os.system(f'cp {file} {composite_mask_path}')
    print("[INFO] Copied video masks to composite")
    
    for file in video_path.glob('colmap_masks/*.png'):
        os.system(f'cp {file} {composite_colmap_path}')
    print("[INFO] Copied video colmap masks to composite")
    
    if (video_path / 'inpainted_frames').exists():
        for file in video_path.glob('inpainted_frames/*.png'):
            os.system(f'cp {file} {composite_inpaint_path}')
        print("[INFO] Copied inpainted video frames to composite")
    else:
        for file in video_path.glob('frames/*.png'):
            os.system(f'cp {file} {composite_inpaint_path}')
        print("[INFO] No inpainted video frames found. Copied original video frames to inpaint.")
    
    if (video_path / 'foreground_masks').exists():
        for file in video_path.glob('foreground_masks/*.png'):
            os.system(f'cp {file} {composite_foreground_mask_path}')
        print("[INFO] Copied foreground masks to composite")
    else:
        print("[INFO] No foreground masks found. Skipping copy.")

    # ------------------------ 2. copy background files to composite ------------------------
    for file in bg_path.glob('images/*.png'):
        os.system(f'cp {file} {composite_img_path}')
    print("[INFO] Copied background images to composite")
    
    for file in bg_path.glob('actor_masks/*.png'):
        os.system(f'cp {file} {composite_mask_path}')
    print("[INFO] Copied background masks to composite")
    
    for file in bg_path.glob('colmap_masks/*.png'):
        os.system(f'cp {file} {composite_colmap_path}')
    print("[INFO] Copied background colmap masks to composite")
    
    if (bg_path / 'inpainted_images').exists():
        for file in bg_path.glob('inpainted_images/*.png'):
            os.system(f'cp {file} {composite_inpaint_path}')
        print("[INFO] Copied inpainted background images to composite")
    else:
        for file in bg_path.glob('images/*.png'):
            os.system(f'cp {file} {composite_inpaint_path}')
        print("[INFO] No inpainted background images found. Copied original background images to inpaint.")
        
    if (bg_path / 'foreground_masks').exists():
        for file in bg_path.glob('foreground_masks/*.png'):
            os.system(f'cp {file} {composite_foreground_mask_path}')
        print("[INFO] Copied foreground masks to composite")
    else:
        print("[INFO] No foreground masks found. Skipping copy.")
    
    # ------------------------ 3. run glomap ------------------------
    
    print("[INFO] Start feature extraction.")
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpus} colmap feature_extractor "
        f"--ImageReader.camera_model PINHOLE "
        f"--ImageReader.mask_path {composite_colmap_path} "
        f"--database_path {composite_path}/database.db "
        f"--image_path {composite_img_path}"
    )
    subprocess.run(cmd, shell=True)
    time.sleep(1) # prevent colmap from collapsing
    print("[INFO] Finished feature extraction.")
    
    print("[INFO] Start feature matching.")
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpus} colmap exhaustive_matcher "
        f"--database_path {composite_path}/database.db"
    )
    subprocess.run(cmd, shell=True)
    time.sleep(1)
    print("[INFO] Finished feature matching.")
    
    print("[INFO] Start feature mapping.")
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpus} glomap mapper "
        f"--database_path {composite_path}/database.db "
        f"--image_path {composite_img_path} "
        f"--output_path {composite_path}/distorted/sparse"
    )
    subprocess.run(cmd, shell=True)
    time.sleep(1)
    print("[INFO] Finished feature mapping.")
    
    print("[INFO] Start image undistortion.") # 3DGS handles pinhole or simple_pinhole format
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpus} colmap image_undistorter "
        f"--image_path {composite_img_path} "
        f"--input_path {composite_path}/distorted/sparse/0 "
        f"--output_path {composite_path}/undistorted "
        f"--output_type COLMAP"
    )
    subprocess.run(cmd, shell=True)
    time.sleep(1)
    print("[INFO] Finished image undistortion.")

    for file in composite_path.glob('undistorted/sparse/*.bin'):
        os.system(f'mv {file} {composite_path}/undistorted/sparse/0')

    print("[INFO] Start generating text outputs.")
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpus} colmap model_converter "
        f"--input_path {composite_path}/undistorted/sparse/0 "
        f"--output_path {composite_path}/undistorted/sparse/0 "
        f"--output_type TXT"
    )
    subprocess.run(cmd, shell=True)
    print("[INFO] Finished generating text outputs.")

    # ------------------------ 4. run depth ------------------------
    print("[INFO] Start generating aligned depths.")
    skip_sfm_opt = "--skip_sfm" if cfg.skip_sfm else ""
    skip_mono_opt = "--skip_mono" if cfg.skip_mono else ""
    visualize_opt = "--visualize" if cfg.visualize else ""
    
    print("[INFO] Start generating aligned inpainted depths.")
    cmd = (
        f"conda run --no-capture-output -n sm3r "
        f"CUDA_VISIBLE_DEVICES={gpus} python -u -m preprocess.align_depths_ddp "
        f"--data_dir {composite_path} "
        f"--image_type inpainted "
        f"{skip_sfm_opt} "
        f"{skip_mono_opt} "
        f"{visualize_opt} "
        f"--gpus {gpus}"
    )
    run_command(cmd)
    print("[INFO] Finished generating aligned inpainted depths.")

    print("[INFO] Start generating aligned original depths.")
    cmd = (
        f"conda run --no-capture-output -n sm3r "
        f"CUDA_VISIBLE_DEVICES={gpus} python -u -m preprocess.align_depths_ddp "
        f"--data_dir {composite_path} "
        f"--image_type original "
        "--skip_sfm "
        f"{skip_mono_opt} "
        f"{visualize_opt} "
        f"--gpus {gpus}"
    )
    run_command(cmd)
    print("[INFO] Finished generating aligned original depths.")
    
if __name__ == "__main__":
    main(tyro.cli(PrepConfig))
