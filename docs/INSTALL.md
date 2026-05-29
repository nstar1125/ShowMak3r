# Installation
We encourage CUDA 11.8, ubuntu 22.04 settings to run ShowMak3r.
## 1. ShowMak3r
### 1.1. Main Environment 
First create main Conda environment for preprocessing and training ShowMak3r.
```bash
# create env
conda create -n sm3r python=3.10 -y
conda activate sm3r
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
```
Install python dependencies. To download chumpy==0.71, you may need to install from source file.
```bash
# install dependencies
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install git+https://github.com/mattloper/chumpy
```
### 1.2. MMPose
Install mmpose for keypoint detection. For more information, you can follow https://mmpose.readthedocs.io/en/latest/installation.html.
```bash
pip install -U openmim
mim install mmengine==0.10.6
mim install mmcv==2.1.0
mim install mmdet==3.2.0
mim install mmpose==1.3.2
```
### 1.3. COLMAP and GLOMAP
Install COLMAP and GLOMAP and for estimating camera parameters. Especially for installing GLOMAP, you need to install latest cmake manually. For more information, you can follow 
[https://github.com/colmap/colmap](https://colmap.github.io/install.html) (COLMAP),
https://github.com/colmap/glomap (GLOMAP).
### 1.4. 3D Gaussian Splatting
Install diff-gaussian-rasterization and simple-knn module for Gaussian Splatting process.
```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```
## 2. Grounded-SAM
Install Grounded-SAM for segmenting actors. We create a new environment for reproductibility. For more information, you can follow https://github.com/IDEA-Research/Grounded-Segment-Anything.

Create new Conda environment for Grounded-SAM.
```bash
# create env
cd submodules/Grounded-Segment-Anything
conda create -n grounded-sam python=3.8
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```
Install python dependencies and download weights.
``` bash
# install dependencies
python3 -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip install -q -r requirements.txt
pip install cmake lit
# download weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../..
```
## 3. CoMotion
Install CoMotion for estimating 3D human mesh. We create a new environment for reproductibility. For more information, you can follow https://github.com/apple/ml-comotion.

Create new Conda environment for CoMotion and install python dependencies.
``` bash
# create env
cd submodules/ml-comotion
conda create -n comotion -y python=3.10
conda activate comotion
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# install dependencies
pip install -e '.[all]' 
```
Download pretrained weights for CoMotion.
``` bash
# download weights
bash get_pretrained_models.sh 
cd ../..
```
Download neutral SMPL body model(version 1.1.0) from <a href="https://smpl.is.tue.mpg.de">SMPL website</a> and follow the provided instructions. After downloading, copy 'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl' to 'src/comotion_demo/data/smpl/SMPL_NEUTRAL.pkl'.

SMPL body model will be utilized for both CoMotion and ShowMak3r.

## 4. Object-Clear
Install Object-Clear for inpainting actors and foreground objects. We create a new environment for reproductibility. For more information, you can follow https://github.com/zjx0101/ObjectClear.

Create new Conda environment for CoMotion and install python dependencies.
``` bash
# create env
cd submodules/ObjectClear
cd ObjectClear
conda create -n objectclear python=3.10 -y
conda activate objectclear
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# install dependencies
pip3 install -r requirements.txt
cd ../..
```