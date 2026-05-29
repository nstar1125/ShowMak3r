from dataclasses import dataclass

@dataclass
class PositionConfig:
    data_path: str
    model_path: str
    gpu: int = 0
    skip_fitting: bool = False
    ## visualization
    render_init_smpl: bool = False
    render_final_smpl: bool = True
    use_wandb: bool = True
    wandb_project: str = "ShowMak3r_position"
    wandb_name = None
    plot_interval: int = 10
    ## parameters
    stage1_iterations: int = 6000
    stage2_iterations: int = 1000
    delete_range: int = 1 # 3
    joint_threshold: float = 0.3
    min_jnts_num: int = 3
    pseudo_gt_conf: float = 0.3
    sample_num: int = 10
    match_threshold: float = 25.0
    smooth_alpha: float = 0.5
    ## fitting loss
    lambda_reproj: float = 1.0 # 1.0
    lambda_depth: float = 1.0 # 1.0
    lambda_traj: float = 0.5 # 0.5
    lambda_contact: float = 0.001 # 0.001