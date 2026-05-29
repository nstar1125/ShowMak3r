import tyro
import torch
from pathlib import Path

from config.pos import PositionConfig
from showmak3r.utils.system_utils import searchForMaxIteration
from showmak3r.utils.io_utils import load_ply
from showmak3r.pipeline.dataset.position_loader import prepare_data
from showmak3r.pipeline.smpl_deform.smpl_wrapper import SMPLWrapper
from showmak3r.vis.viser_smpl import visualize_init_scene

def main(cfg: PositionConfig):
    data_path = Path(cfg.data_path)
    model_path = Path(cfg.model_path)

    loaded_data = prepare_data(data_path, model_path)
    optimized_result = loaded_data["associated"]

    # # plot 3D trajectory
    # if True:
    #     import matplotlib.pyplot as plt
    #     import numpy as np

    #     for pnum, person_dict in optimized_result[0].items():
    #         trans_list = []
    #         for fname, frame_dict in person_dict.items():
    #             smpl_param = person_dict[fname]['smpl_param'].squeeze().float()
    #             trans_list.append(smpl_param[1:4].numpy())
    #         trajectory = np.stack(trans_list, axis=0)
            
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, projection='3d')
    #         # ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='3D Trajectory', color='blue')
    #         ax.plot(trajectory[:, 0], trajectory[:, 2], -trajectory[:, 1], label='3D Trajectory', color='blue')
    #         # ax.set_xlabel('X')
    #         # ax.set_ylabel('Y')
    #         # ax.set_zlabel('Z')
    #         ax.set_xlabel('X')
    #         ax.set_ylabel('Z')
    #         ax.set_zlabel('-Y')
            
    #         plt.title('w/ trajectory loss')
    #         plt.legend()
    #         plt.savefig(str(model_path / "actor" / f"{pnum:03d}" / "trajectory.png"))
    #         plt.close()
    # # exit()

    img_dict = loaded_data["images"]
    colmap_camdicts = loaded_data["colmap"]
    scene_pcd = loaded_data["sfm_points"]
    device = torch.device(f"cuda:{cfg.gpu}")
    
    smpl_model = SMPLWrapper("submodules/ml-comotion/src/comotion_demo/data/smpl", use_feet_keypoints=True, device=device)

    pcd_path = model_path / "stage" / "simple_pcd"
    max_iter = searchForMaxIteration(pcd_path)
    if pcd_path.exists():
        xyz, rgb = load_ply(pcd_path)
        scene_pcd = (xyz, rgb)

    print("[INFO] Start viser visualization.")
    visualize_init_scene(
        optimized_result[0], # associated dict has only one shot
        smpl_model,
        scene_pcd,
        colmap_camdicts,
        img_dict,
        device
    )
    
if __name__ == "__main__":
    main(tyro.cli(PositionConfig))