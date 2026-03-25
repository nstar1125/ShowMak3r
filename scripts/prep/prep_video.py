import tyro
import pathlib
import os
import subprocess
from config.prep import PrepConfig

# run command with subprocess
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
    if len(data.split("/")) > 1:
        data_name = data.split("/")[-1]
    else:
        data_name = data
    gpus = cfg.gpus
    
    data_path = pathlib.Path(os.path.join("demo",data)) # input path
    sam_path = pathlib.Path("submodules/Grounded-Segment-Anything") # sam path
    
    background_path = data_path / f'background'
    bg_img_path = background_path / "images"
    
    video_path = data_path / f'video'
    video_path.mkdir(exist_ok=True)
    
    frame_path = video_path / 'frames'
    frame_path.mkdir(exist_ok=True)
    
    actor_mask_path = video_path / 'actor_masks'
    actor_mask_path.mkdir(exist_ok=True)

    merge_mask_path = video_path / 'merged_masks'
    merge_mask_path.mkdir(exist_ok=True)
    
    colmap_mask_path = video_path / 'colmap_masks'
    colmap_mask_path.mkdir(exist_ok=True)

    inpaint_path = video_path / 'inpainted_frames'
    inpaint_path.mkdir(exist_ok=True)

    # ------------------------ 1. extract frames from video ------------------------
    print("[INFO] Start extracting frames from videos.")
    # calculate frame width and height
    
    if bg_img_path.exists():
        first_image = next(bg_img_path.glob('*.png'), None)
        cmd = (
                f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 {first_image}"
            )
        result = subprocess.check_output(cmd, shell=True).decode()
        width, height = map(int, result.strip().split('x'))
        print(f"[INFO] Using background image size: {width}x{height}")
    else:
        cmd = (
            f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 {video_path}/{data_name}.mp4"
        )
        result = subprocess.check_output(cmd, shell=True).decode() 
        width, height = map(int, result.strip().split('x'))
        print(f"[INFO] Using video frame size: {width}x{height}")
    
    # extract frames from video
    cmd = (
        f"ffmpeg -i {video_path}/{data_name}.mp4 -vf scale={width}:{height} -r {cfg.fps} {frame_path}/frame_%04d.png"
    )
    subprocess.run(cmd, shell=True)
    
    # detect shot change
    cmd = (
        f"ffmpeg -i {video_path}/{data_name}.mp4 "
        f"-vf \"select='gt(scene,{cfg.shot_threshold})',showinfo,scale={width}:{height}\" "
        "-vsync cfr -f null - 2>&1 | "
        f"sed -n 's/.*pts_time:\([^ ]*\).*/\\1/p' | "
        f"awk -v fps={cfg.fps} '{{print int($1 * fps)}}' > {video_path}/shot_change.log"
    ) # TODO - fix shot detection error in low fps
    subprocess.run(cmd, shell=True)
    print("[INFO] Finished extracting frames from videos.")

    # ------------------------ 2. generate video masks ------------------------
    print("[INFO] Start generating video masks.")
    cmd = (
        f"conda run --no-capture-output -n grounded-sam "
        f"CUDA_VISIBLE_DEVICES={gpus} python -u preprocess/grounded_sam_ddp.py "
        f"--config {sam_path}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py "
        f"--grounded_checkpoint {sam_path}/groundingdino_swint_ogc.pth "
        f"--sam_checkpoint {sam_path}/sam_vit_h_4b8939.pth "
        f"--input_dir {frame_path} "
        f"--output_dir {actor_mask_path} "
        f"--text_prompt human "
        f"--box_threshold {cfg.box_threshold} "
        f"--gpus {gpus} "
    )
    run_command(cmd)
    print("[INFO] Finished generating video masks.")

    # ------------------------ 3. extract frames from foreground video ------------------------
    if os.path.exists(video_path/f'mask_{data_name}.mp4'):
        fg_mask_path = video_path / 'foreground_masks'
        if not os.path.exists(fg_mask_path):
            os.mkdir(fg_mask_path)
        cmd = (
            f"ffmpeg -i {video_path}/mask_{data_name}.mp4 -vf scale={width}:{height} -r {cfg.fps} {fg_mask_path}/frame_%04d.png"
        )
        subprocess.run(cmd, shell=True)
        
        num_fg = len(list(fg_mask_path.glob('frame_*.png')))
        num_img = len(list(frame_path.glob('frame_*.png')))
        if num_img == num_fg:
            pass
        elif num_img - num_fg == 1 or num_img - num_fg == 2:
            print("[INFO] adjusting foreground mask frame numbers.")
            last_frame = f"{num_fg + 1:04d}"
            cmd = (
                f"ffmpeg -sseof -0.05 -i {video_path}/mask_{data_name}.mp4 -vf scale={width}:{height} "
                f"-r {cfg.fps} -vsync vfr -frames:v 1 {fg_mask_path}/frame_{last_frame}.png"
            )
            subprocess.run(cmd, shell=True)
            if num_img - num_fg == 2:
                last_frame = f"{num_fg + 2:04d}"
                cmd = (
                    f"ffmpeg -sseof -0.05 -i {video_path}/mask_{data_name}.mp4 -vf scale={width}:{height} "
                    f"-r {cfg.fps} -vsync vfr -frames:v 1 {fg_mask_path}/frame_{last_frame}.png"
                )
                subprocess.run(cmd, shell=True)
            print(f"[INFO] added frame_{last_frame}.png to the foreground_masks")
        elif num_fg - num_img == 1 or num_fg - num_img == 2:
            print("[INFO] adjusting foreground mask frame numbers.")
            remove_frame = f"{num_fg:04d}"
            cmd = (
                f"rm {fg_mask_path}/frame_{remove_frame}.png"
            )
            subprocess.run(cmd, shell=True)
            if num_fg - num_img == 2:
                remove_frame = f"{num_fg-1:04d}"
                cmd = (
                    f"rm {fg_mask_path}/frame_{remove_frame}.png"
                )
                subprocess.run(cmd, shell=True)
            print(f"[INFO] removed frame_{remove_frame}.png from the foreground_masks")
        else:
            print("[INFO] adjust foreground masks video length.")
            exit(1)
    
    # ------------------------ 4. generate colmap masks ------------------------
    if os.path.exists(video_path/f'mask_{data_name}.mp4'):
        print("[INFO] Start generating colmap masks with foreground.")
        cmd = (
            f"conda run --no-capture-output -n sm3r "
            f"CUDA_VISIBLE_DEVICES={gpus} python -u preprocess/colmap_masks.py "
            f"--input_paths {actor_mask_path} {fg_mask_path} "
            f"--merge_path {merge_mask_path} "
            f"--output_path {colmap_mask_path} "
        )
    else:
        print("[INFO] Start generating colmap masks.")
        cmd = (
            f"conda run --no-capture-output -n sm3r "
            f"CUDA_VISIBLE_DEVICES={gpus} python -u preprocess/colmap_masks.py "
            f"--input_paths {actor_mask_path} "
            f"--merge_path {merge_mask_path} "
            f"--output_path {colmap_mask_path} "
        )
    run_command(cmd)
    print("[INFO] Finished generating colmap masks.")

    # ------------------------ 5. inpaint video frames ------------------------
    print("[INFO] Start inpainting video foreground.")
    cmd = (
        f"conda run --no-capture-output -n sm3r "
        f"CUDA_VISIBLE_DEVICES={gpus} python -u preprocess/objectclear_ddp.py "
        f"--input_path {frame_path} "
        f"--mask_path {merge_mask_path} "
        f"--output_path {inpaint_path} "
        f"--gpus {gpus} "
    )
    run_command(cmd)
    print("[INFO] Finished inpainting video foreground.")

    # ------------------------ 6. estimate 3D poses ------------------------
    print("[INFO] Start estimating human 3D poses.")
    single_gpu = gpus.split(',')[0] # use the first gpu of the list.
    boundary_list = [0]
    with open(video_path / "shot_change.log", 'r') as file:
        for line in file:
            line = line.strip()
            frame_boundary = int(line)
            boundary_list.append(frame_boundary)
    boundary_list.append(len(list(frame_path.glob("*.png"))))
    
    for shot_id in range(len(boundary_list)-1):
        start_fnum = boundary_list[shot_id]
        end_fnum = boundary_list[shot_id+1]
        print(f"[INFO] Estimating human 3D poses in shot {shot_id} (frame_{start_fnum+1:04d} to frame_{end_fnum:04d}).")
        cmd = (
            f"PYTHONPATH=$PYTHONPATH:submodules/ml-comotion "
            f"conda run --no-capture-output -n comotion "
            f"CUDA_VISIBLE_DEVICES={single_gpu} python -u -m demo "
            f"--input-path {frame_path} "
            f"--output-dir {video_path}/comotion/shot_{shot_id} "
            f"--start-frame {start_fnum} "
            f"--num-frames {end_fnum-start_fnum} "
        )
        run_command(cmd)
    print("[INFO] Finished estimating human 3D poses.")
    
    # ------------------------ 7. process 3d pose result ------------------------
    print("[INFO] Start processing comotion result format.")
    cmd = (
        f"conda run --no-capture-output -n comotion "
        f"CUDA_VISIBLE_DEVICES={gpus} python -u -m preprocess.process_comotion "
        f"--data_dir {video_path} "
    )
    run_command(cmd)
    print("[INFO] Finished processing comotion result format.")
    
    # ------------------------ 8. estimate 2D poses ------------------------
    print("[INFO] Start estimating human 2D poses.")
    cmd = (
        f"conda run --no-capture-output -n sm3r "
        f"CUDA_VISIBLE_DEVICES={gpus} python -u -m preprocess.dwpose_ddp "
        f"--data_dir {video_path} "
        f"--visualize "
        f"--gpus {gpus} "
        f"--conf_threshold {cfg.conf_threshold} "
        f"--boundary_weight {cfg.boundary_weight} "
        f"--boundary_length {cfg.boundary_length} "
        )
    run_command(cmd)
    print("[INFO] Finished estimating human 2D poses.")
    
    # ------------------------ 9. merge tracklets ------------------------
    print("[INFO] Start merging 3d and 2d pose results.")
    cmd = (
            f"conda run --no-capture-output -n sm3r "
            f"CUDA_VISIBLE_DEVICES={gpus} python -u -m preprocess.merge_2d3d "
            f"--data_dir {video_path} "
            )
    run_command(cmd)
    print("[INFO] Finished merging 3d and 2d pose results.")

    # ------------------------ 10. personal masks ------------------------
    print("[INFO] Start personal segmentation.")
    cmd = (
            f"conda run --no-capture-output -n grounded-sam "
            f"CUDA_VISIBLE_DEVICES={gpus} python -u preprocess/person_segment_ddp.py "
            f"--data_dir {video_path} "
            f"--sam_checkpoint {sam_path}/sam_vit_h_4b8939.pth "
            f"--gpus {gpus} "
            )
    run_command(cmd)
    print("[INFO] Finished personal segmentation.")

if __name__ == "__main__":
    main(tyro.cli(PrepConfig))