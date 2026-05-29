# ShowMak3r: Compositional TV Show Reconstruction
<div align="center">
    <a href="https://nstar1125.github.io" target='_blank'>Sangmin Kim</a>&emsp;
    <a href='https://www.linkedin.com/in/seunguk-do-9b42251a9' target='_blank'>Seunguk Do</a>&emsp;
    <a href='https://jaesik.info' target='_blank'>Jaesik Park</a>&emsp;
</div>
<div align="center">
    Seoul National University
</div>

<div>
    <h4 align="center">
        <a href="https://nstar1125.github.io/showmak3r/" target='_blank'>
        <img src="https://img.shields.io/badge/🎬-Project%20Page-blue">
        </a>
        <a href="https://arxiv.org/abs/2504.19584" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2504.19584-b31b1b.svg">
        </a>
    </h4>
</div>

**TL;DR** We reconstruct dynamic radiance fields from TV shows, enabling editing of the scenes like how video clips are made in a production control room.


<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="assets/figure_teaser.png">
</div>

## Installation
First clone ShowMak3r repository.
```bash
# clone repo
git clone https://github.com/nstar1125/ShowMak3r.git --recursive
cd ShowMak3r
```
Next, follow <a href="docs/INSTALL.md">INSTALL.md</a> to set up all Conda environments.
## Preprocessing
To run ShowMak3r, data must be placed like the following directory. Download demo data from https://drive.google.com/drive/folders/1FyspKVDX5W8pqBXh7m1un_QAqY-m9LFI?usp=sharing, and place them under 'demo'.
```bash
<video_name>
├── background
│   ├── images
│   │   ├── <bg_img_1>.png
│   │   └── ...
│   └── masks #optional
│       ├── <bg_img_1>.png
│       └── ...
└── video
    ├── mask_<video_name>.mp4 #optional
    └── <video_name>.mp4
```
After setting up the data directory, run the provided preprocessing script as following:
```bash
# preprocessing overall
bash scripts/run_prep.sh {DATA_NAME} {GPUS}
```
or
```bash
# preprocessing background, and video files
python -m scripts.prep.prep_stage --data {DATA_NAME} --gpus {GPUS}
python -m scripts.prep.prep_video --data {DATA_NAME} --gpus {GPUS}
python -m scripts.prep.prep_composite --data {DATA_NAME} --gpus {GPUS}
```
For detailed instructions, follow <a href="docs/PREP.md">PREP.md</a>.
## Training
To run ShowMak3r, follow the commands below in order.
1. After preprocessing data, train static 3D stage by running the script below.
    ```bash
    # train 3D stage
    bash scripts/train_stage.sh {DATA_NAME} {GPU}
    ```
2. After training 3D stage, position actor SMPLs to the stage coordinate system by running the script below.
    ```bash
    # position actor SMPLs
    bash scripts/train_position.sh {DATA_NAME} {GPU}
    ```
3. After positioning actor SMPLs, first train Custom Diffusion model for SDS loss. Then, train 3D actors by running the script below.
    ```bash
    # train Custom Diffusion model
    bash scripts/train_diffusion.sh {DATA_NAME} {GPU}
    # train 3D actors
    bash scripts/train_actor.sh {DATA_NAME} {EXP_NAME} {GPU}
    ```
## Visualization
### Stage
After training 3D stage, you can visualize the static stage with Viser API by running the script below.
```bash
# visualize 3D stage
bash scripts/test_stage.sh {DATA_NAME} {GPU}
```
### Position
After positioning actor SMPLs, you can visualize aligned actor SMPLs and the stage point cloud with Viser API by running the script below.
```bash
# visualize SMPLs and the stage point cloud
bash scripts/test_position.sh {DATA_NAME} {GPU}
```
### Actors
After training 3D actors, you can edit or visualize 4D scene with Viser API by running the script below.
```bash
# visualize 4D scene
bash scripts/test_actor.sh {DATA_NAME} {EXP_NAME} {GPU}
```
## Acknowledgement
Our code is mainly based on 'Guess The Unseen: Dynamic 3D Scene Reconstruction from Partial 2D Glimpses (CVPR 2024)'. 

These are the codes we referenced for our project. Check out their awesome works if you are intereseted.
- [Guess The Unseen: Dynamic 3D Scene Reconstruction from Partial 2D Glimpses](https://github.com/snuvclab/gtu?tab=readme-ov-file)
- [Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction](https://github.com/ingra14m/Deformable-3D-Gaussians)
- [Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images](https://github.com/robot0321/DepthRegularizedGS)
## Citation
If you find our repo useful for your research, please consider citing our paper:
```
@article{kim2025showmak3r,
  author    = {Kim, Sangmin and Do, Seunguk and Park, Jaesik},
  title     = {ShowMak3r: Compositional TV Show Reconstruction},
  journal   = {CVPR},
  year      = {2025}
}
```