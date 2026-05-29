from dataclasses import dataclass

# global options
BOX_THRESHOLD = 0.25

@dataclass
class PrepConfig:
    data: str
    gpus: str = '0'
    # grounded sam
    box_threshold: float = BOX_THRESHOLD
    # ffmpeg
    fps: int = 30 
    shot_threshold: float = 0.4 # 0.25
    # dwpose
    conf_threshold: float = 0.35 # 0.5
    boundary_weight: float = 0.5
    boundary_length: float = 0.1
    # depth alignment
    skip_sfm: bool = False
    skip_mono: bool = False
    visualize: bool = True