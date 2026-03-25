import torch
import tyro
import cv2
import numpy as np
import open3d as o3d
import trimesh
import wandb
import shutil
import pickle
import time
from pathlib import Path
from tqdm import tqdm
from config.pos import PositionConfig
from showmak3r.utils.io_utils import write_pickle
from showmak3r.utils.camera_utils import estimate_translation_cv2
from showmak3r.utils.image_utils import draw_bbox_w_pid, draw_bodypose_with_color
# from showmak3r.utils.system_utils import format_time
from showmak3r.pipeline.smpl_deform.smpl_wrapper import SMPLWrapper
from showmak3r.pipeline.renderer.renderer_wrapper import render_smpl_full
from showmak3r.pipeline.positioning.locator3d import fit_single_actor
from showmak3r.pipeline.positioning.shot_matcher import associate_actors
from showmak3r.pipeline.dataset.position_loader import prepare_data

def main(cfg: PositionConfig):
    data_path = Path(cfg.data_path)
    model_path = Path(cfg.model_path)
    device = torch.device(f"cuda:{cfg.gpu}")
    
    actor_path = model_path / "actor"
    actor_path.mkdir(parents=True, exist_ok=True)

    # prepare data
    loaded_data = prepare_data(data_path, model_path)
    merged_result = loaded_data["merged"]
    aligned_result = loaded_data["aligned"]
    img_dict = loaded_data["images"]
    mask_dicts = loaded_data["masks"]
    depth_dicts = loaded_data["depths"]
    colmap_camdicts = loaded_data["colmap"]
    boundary_list = [0]
    boundary_list.extend(loaded_data["boundary"])
    scene_pcd = loaded_data["sfm_points"]
    stage_depth_dict = loaded_data["stage_depths"]

    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_name)
    
    smpl_model = SMPLWrapper("submodules/ml-comotion/src/comotion_demo/data/smpl", use_feet_keypoints=True, device=device)

    if aligned_result is not None and cfg.skip_fitting:
        merged_result = aligned_result.copy()
    else:
        print("[INFO] Start converting SMPL parameters from camera space to world space.")
        for fname, colmap_camdict in colmap_camdicts.items():
            # get default comotion projection matrix
            res = img_dict[fname].shape[:2]
            max_res = max(res)
            comotion_intr = np.array([[2 * max_res, 0, 0.5 * res[1], 0], 
                          [0, 2 * max_res, 0.5 * res[0], 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype=np.float32)
            comotion_extr = np.eye(4)
            comotion_proj = comotion_intr @ comotion_extr
            
            # get colmap camera intrinsic
            new_intrinsic = colmap_camdict['intrinsic'][:3,:3] 
            
            for shot_id, shot_dict in merged_result.items(): # per shot
                for pnum, person_dict in shot_dict.items(): # per person       
                    if (fname in person_dict) and (person_dict[fname]['smpl_param'] is not None):
                        if person_dict[fname]['smpl_param'][0] != 1:
                            raise NotImplementedError("We currently do not consider the initial SMPL scale != 1")
                        
                        if fname not in depth_dicts[shot_id][pnum]: # skip if no depth map
                            continue

                        # get initial distance
                        depth_map = depth_dicts[shot_id][pnum][fname]
                        background_mask = depth_map!=np.inf
                        if depth_map[background_mask].size > 0:
                            initial_distance = depth_map[background_mask].mean()
                        else:
                            initial_distance = None
                        
                        # get smpl joints in projective space 
                        smpl_param = person_dict[fname]['smpl_param'].to(device).float().squeeze().reshape(1, -1) # (1, 86)
                        smpl_output = smpl_model(smpl_param)
                        smpl_jnts = smpl_output['smpl_jnts'].squeeze().detach().cpu() # (J,3)
                        smpl_jnts = smpl_jnts[:25] # Only consider SMPL-25 skeletons here (25,3)
                        smpl_jnts_homo = torch.cat([smpl_jnts, torch.ones_like(smpl_jnts[:, :1])], dim=-1) # to homogeneous coordinates (25,4)
                        smpl_jnts_homo = smpl_jnts_homo.numpy()
                        original_pj_jnts = np.einsum('ij,bj->bi', comotion_proj, smpl_jnts_homo) # apply projection to each joints (25,4)
                        original_pj_jnts = original_pj_jnts[:, :2] / (original_pj_jnts[:, 2:3] + 1e-9) # to projective space

                        # get translation and rotation in camera space with PnP
                        translation_pred, rotation_pred = estimate_translation_cv2(joints_3d=smpl_jnts.numpy(), joints_2d=original_pj_jnts, proj_mat=new_intrinsic)
                        pnp_translation = translation_pred.reshape(-1).astype(np.float32)
                        pnp_rotation = cv2.Rodrigues(rotation_pred)[0].astype(np.float32) # (3, 3) -> (3,)

                        # get neutral SMPL pelvis translations
                        neutral_param = torch.zeros(1, 86).float().to(device)
                        neutral_param[0,0] = 1
                        neutral_param[0,-10:] = smpl_param.squeeze()[-10:] # copy beta
                        neutral_smpl_output = smpl_model(neutral_param)
                        pelvis_translation = neutral_smpl_output['smpl_jnts'].squeeze().detach().cpu()[0].numpy()
                        
                        # apply translation and rotation in camera space to the regressed SMPL global parameters
                        smpl_pose = person_dict[fname]['smpl_param'][4:-10].numpy().astype(np.float32) # (72,)
                        reg_smpl_rotation = cv2.Rodrigues(smpl_pose[:3])[0].astype(np.float32)
                        reg_smpl_translation = person_dict[fname]['smpl_param'][1:4].numpy().astype(np.float32)

                        cam_smpl_rotation = pnp_rotation @ reg_smpl_rotation # smpl in camera space
                        cam_smpl_translation = (pnp_rotation @ (reg_smpl_translation + pelvis_translation) + pnp_translation) - pelvis_translation # move to origin and rotate and move back
                        
                        # # rescale with initial distance
                        # if initial_distance is not None:
                        #     initial_scale = initial_distance / cam_smpl_translation[-1]
                        # else:
                        #     initial_scale = 1
                        initial_scale = 1
                        cam_smpl_translation[-1] *= initial_scale

                        # convert camera-space SMPL global parameters to world-space
                        R = colmap_camdict['w2c'][:3, :3]
                        T = colmap_camdict['w2c'][:3, -1]
                        w2c_rotation = R.astype(np.float32)
                        w2c_translation = T.reshape(-1).astype(np.float32)

                        world_smpl_rotation = w2c_rotation.T @ cam_smpl_rotation # apply c2w = cam-to-world space
                        world_smpl_translation = w2c_rotation.T @ (pelvis_translation + cam_smpl_translation - w2c_translation) - pelvis_translation

                        # convert to parameter shape
                        world_smpl_rot_vec = cv2.Rodrigues(world_smpl_rotation)[0] # (3,)
                        smpl_pose[:3] = world_smpl_rot_vec.reshape(-1)
                        smpl_trans = world_smpl_translation

                        # collect new SMPL parameters
                        new_smpl_param = person_dict[fname]['smpl_param']
                        new_smpl_param[0] *= torch.tensor(initial_scale)
                        new_smpl_param[1:4] = torch.tensor(smpl_trans)
                        new_smpl_param[4:-10] = torch.tensor(smpl_pose)
                        person_dict[fname]['smpl_param'] = new_smpl_param

    # render initial smpls
    if cfg.render_init_smpl:
        print("[INFO] Start rendering initial SMPLs.")
        init_render_path = actor_path / "initial_render"
        init_render_path.mkdir(parents=True, exist_ok=True)
        render_smpl_full(
            cfg,
            init_render_path,
            "initial_render",
            img_dict,
            colmap_camdicts,
            merged_result,
            smpl_model,
            device
        )
        print("[INFO] Finished rendering initial SMPLs.")
    
    # ================================================================================
    #                                  3DLocator
    # ================================================================================
    time_stamps = []
    time_stamps.append(time.time())
    if not cfg.skip_fitting:
        print("[INFO] Start fitting SMPL parameters to the scene.")

        shot_dicts = merged_result.copy()
        for shot_id, shot_dict in shot_dicts.items():
            for pnum in list(shot_dict.keys()):
                person_dict = shot_dict[pnum]
                
                start_fid = boundary_list[shot_id]
                num_frames = boundary_list[shot_id+1] - start_fid
                
                img_dict_batch = dict()
                mask_dict_batch = dict()
                stage_depth_dict_batch = dict()
                colmap_camdicts_batch = dict()
                for fid in range(start_fid, start_fid+num_frames):
                    fname = f"frame_{fid+1:04d}"
                    if fname in img_dict and fname in mask_dicts[shot_id][pnum]:
                        img_dict_batch[fname] = img_dict[fname]
                        mask_dict_batch[fname] = mask_dicts[shot_id][pnum][fname]
                        stage_depth_dict_batch[fname] = stage_depth_dict[fname]
                        colmap_camdicts_batch[fname] = colmap_camdicts[fname]
                
                depth_dict_batch = depth_dicts[shot_id][pnum]

                person_dict = fit_single_actor(
                    cfg, pnum, start_fid, num_frames,
                    person_dict, 
                    colmap_camdicts_batch, 
                    img_dict_batch,
                    mask_dict_batch,
                    depth_dict_batch,
                    stage_depth_dict_batch,
                    smpl_model,
                    device
                )
                if len(person_dict) > 0:
                    merged_result[shot_id][pnum] = person_dict
                else:
                    print(f"[INFO] Not enough valid frames in shot {shot_id}. Deleting person {pnum}.")
                    del merged_result[shot_id][pnum] # remove empty person

        save_path = model_path / "actor" / 'aligned_result.pkl'
        with open(save_path, 'wb') as file:
            pickle.dump(merged_result, file)
        print("[INFO] Saved aligned result.")
    
    time_stamps.append(time.time())
    # ================================================================================
    #                                   ShotMatcher
    # ================================================================================
    map_dict = dict()
    if len(boundary_list) > 2: # if more than one shot, start matching process
        print("[INFO] Start associating actors.")
        merged_result, map_dict = associate_actors(cfg, merged_result, boundary_list)

    save_path = model_path / "actor" / 'associated_result.pkl'
    with open(save_path, 'wb') as file:
        pickle.dump(merged_result, file)
    print("[INFO] Saved associated result.")
        
    time_stamps.append(time.time())
    # ================================================================================
    
    if cfg.use_wandb:
        wandb.finish()
            
    if cfg.render_final_smpl:
        print("[INFO] Start rendering final SMPLs.")
        final_render_path = actor_path / "fitted_render"
        final_render_path.mkdir(parents=True, exist_ok=True)

        render_smpl_full(
            cfg,
            final_render_path,
            "fitted_render",
            img_dict,
            colmap_camdicts,
            merged_result,
            smpl_model,
            device
        )
        print("[INFO] Finished rendering final SMPLs.")

    # =============================== Save Results ===============================
    # save scene cameras
    save_dict = dict()
    for fname, camdict in colmap_camdicts.items():
        w2c_R = camdict['w2c'][:3,:3]
        w2c_T = camdict['w2c'][:3, 3]
        
        save_dict[fname] = dict(
            camera = dict(
                    width = camdict['W'],
                    height = camdict['H'],
                    rotation = w2c_R,
                    translation = w2c_T,
                    intrinsic = camdict['intrinsic'][:3,:3]
                ),
        )
    write_pickle(actor_path / f'cameras.pkl', save_dict)
    print(f"[INFO] Saved scene cameras.")
    
    # copy individual masks
    if len(map_dict) > 0: # more than one shot
        for old_key, new_key in map_dict.items():
            source_dir = list((Path(cfg.data_path) / "video" / "personal_masks").glob(f"*_{old_key:03d}"))[0]
            target_dir = Path(cfg.model_path) / "actor" / f"{new_key:03d}" / "masks"
            target_dir.mkdir(parents=True, exist_ok=True)

            # copy files
            if source_dir.exists():
                for img_file in source_dir.glob("*"):
                    if img_file.is_file() and img_file.suffix.lower() in [".jpg", ".png"]:
                        shutil.copy(img_file, target_dir / img_file.name)
            else:
                assert False
    else: # only one shot
        for pnum in merged_result[0].keys():
            source_dir = list((Path(cfg.data_path) / "video" / "personal_masks").glob(f"*_{pnum:03d}"))[0]
            target_dir = Path(cfg.model_path) / "actor" / f"{pnum:03d}" / "masks"
            target_dir.mkdir(parents=True, exist_ok=True)

            for img_file in source_dir.glob("*"):
                if img_file.is_file() and img_file.suffix.lower() in [".jpg", ".png"]:
                    shutil.copy(img_file, target_dir / img_file.name)
    
    # save individual results
    for i, pnum in enumerate(sorted(list(merged_result[0].keys()))):
        person_dict = merged_result[0][pnum]
        save_dict = dict()

        # save optimized result
        for fname, frame_dict in person_dict.items():
            cam_dict = colmap_camdicts[fname]
            smpl_param = frame_dict['smpl_param']
            w2c_R = cam_dict['w2c'][:3,:3]
            w2c_T = cam_dict['w2c'][:3, 3]
            save_dict[fname] = dict(
                gt_body_pose = frame_dict['body'],
                gt_bbox = frame_dict['bbox'],
                smpl_param = np.expand_dims(smpl_param, axis=0), # (1, 86)
                camera = dict(
                        width = cam_dict['W'],
                        height = cam_dict['H'],
                        rotation = w2c_R,
                        translation = w2c_T,
                        intrinsic = cam_dict['intrinsic'][:3,:3]
                    ),
            )
        write_pickle(actor_path / f"{pnum:03d}" / f'optimized.pkl', save_dict)
        print(f"[INFO] Saved optimized result for person {pnum}.")

        # save initial SMPL 3D GSs (canonical SMPL model)
        smpl_shape = list(save_dict.values())[0]['smpl_param'][:, -10:]
        param_canon = np.concatenate([
                                np.ones( (1,1)) * 1, 
                                np.zeros( (1,3)),
                                np.zeros( (1,72)),
                                smpl_shape], axis=1)
        param_canon[0, 9] = np.pi / 6
        param_canon[0, 12] = -np.pi / 6
        smpl_params = torch.from_numpy(param_canon).to(device).float()
        smpl_output = smpl_model(smpl_params)

        canon_smpl_verts = smpl_output['smpl_verts'].data.cpu().numpy().squeeze() 
        smpl_faces = smpl_model.smpl.faces.astype(np.int64)

        trimesh_mesh = trimesh.Trimesh(vertices=canon_smpl_verts, faces=smpl_faces)
        normals = trimesh_mesh.vertex_normals
        rgbs = np.ones_like(canon_smpl_verts, dtype=np.uint8) * np.array([[128, 128, 128]])   # mean gray color
        
        with open(actor_path / f"{pnum:03d}" / f"canonical.txt", 'w') as f:
            for i, p in enumerate(canon_smpl_verts):
                xyz = p.tolist()
                rgb = rgbs[i].tolist()
                normal = normals[i].tolist()
                error = 0
                f.write(f"{i} {xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]} {error} {normal[0]} {normal[1]} {normal[2]}\n")
        print(f"[INFO] Saved canonical SMPL for person {pnum}.")

if __name__ == "__main__":
    main(tyro.cli(PositionConfig))