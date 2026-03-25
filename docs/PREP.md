# Preprocessing
We provide a few demo examples from various sitcom series like 'The Big Bang Theory', 'Friends', .. etc.
## 1. Data structure
Before preprocessing the data, files must be positioned like the following. Background directory contains files that depicit the background images. You can download background images of Sitcoms3D dataset for demo from https://github.com/ethanweber/sitcoms3D.

Video directory contains the video file that we want to reconstruct. If there are foreground objects that you want to additioanlly erase, use <a href="https://ai.meta.com/sam2/">SAM2 demo</a> to get a masked video where targets are white and the background is black. 

Data structure should look like this:
```bash
<video_name>
├── background
│   ├── images
│   │   ├── <bg_img_1>.png
│   │   └── ...
│   └── masks # optional
│       ├── <bg_img_1>.png
│       └── ...
└── video
    ├── mask_<video_name>.mp4 # optional
    └── <video_name>.mp4
```
## 2. Preprocessing stage
To preprocess background directory, run the command below. Input the name of the video for `--data`, and the GPU IDs for `--gpus`. You can assign multiple GPUs for distributed preprocessing. Seperate the GPU IDs like `0,1,2,3`.
```bash
# preprocessing background files
python -m scripts.prep.prep_background --data {DATA_PATH} --gpus {GPUS}
```
After preprocessing the background directory, data structure should look like this:
```bash
<video_name>
├── background
│   ├── actor_masks
│   ├── colmap_masks
│   ├── foreground_masks # optional
│   ├── images
│   ├── inpainted_images
│   ├── merged_masks
│   └── masks
└── video
```
## 3. Preprocessing video
To preprocess video directory, run the command below. Input the name of the video for `--data`, and the GPU IDs for `--gpus`. You can assign multiple GPUs for distributed preprocessing. Seperate the GPU IDs like `0,1,2,3`.
```bash
# preprocessing video files
python -m scripts.prep.prep_video --data {DATA_PATH} --gpus {GPUS}
```
After preprocessing the video directory, data structure should look like this:
```bash
<video_name>
├── background
│   └── ...
└── video
    ├── actor_masks
    ├── colmap_masks
    ├── comotion
    │   ├── shot_<shot_id>
    │   ├── ...
    │   └── comotion_result.pkl
    ├── dwpose
    │   ├── <shot_id>_<cand_num>
    │   ├── ...
    │   └── dwpose_result.pkl
    ├── foreground_masks
    ├── frames
    ├── inpainted_frames
    ├── merged_masks
    ├── personal_masks
    │   ├── <shot_id>_<cand_num>
    │   └── ...
    ├── mask_<video_name>.mp4 # optional
    ├── merged_result.pkl
    ├── shot_change.log
    └── <video_name>.mp4
```
`comotion` directory contains the results from comotion, which are separated by `shot_change.log`. `dwPose` directory contains the 2D Keypoint results from DWPose which are separated by shot and candidate number. `personal_masks` directory contains the separated masks for each candidate. `shot_change.log` contains the frame number at shot transition.

## 4. Preprocessing composite
After preprocessing the background and video directory, run the command below to merge and preprocess them in composite. Input the name of the video for `--data`, and the GPU IDs for `--gpus`. You can assign multiple GPUs for distributed preprocessing. Seperate the GPU IDs like `0,1,2,3`.
```bash
# preprocessing background and video files
python -m scripts.prep.prep_video --data {DATA_PATH} --gpus {GPUS}
```
After preprocessing them in composite, final data structure should look like this:
```bash
<video_name>
├── background
│   └── ...
├── video
│   └── ...
└── composite
    ├── actor_masks
    ├── colmap_masks
    ├── depths
    │   ├── inpainted_aligned
    │   ├── inpainted_mono
    │   ├── original_aligned
    │   ├── original_mono
    │   └── sfm_depths
    ├── dwpose
    ├── distorted
    ├── foreground_masks
    ├── images
    ├── inpainted_images
    ├── undistorted
    └── database.db
```
`depths` directory contains the depth estimation results from inpainted and original images, which are then aligned to SfM points. `undistorted` directory contains camera parameters estimated from GLOMAP.