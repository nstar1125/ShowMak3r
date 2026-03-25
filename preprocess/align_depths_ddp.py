import numpy as np
import os 
import sys
import torch
import cv2
import multiprocessing as mp
from argparse import ArgumentParser
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from dataclasses import dataclass
from moge.model.v2 import MoGeModel

from preprocess.utils.colmap_parsing_utils import qvec2rotmat, read_cameras_binary, read_cameras_text, read_images_binary, read_images_text, read_points3D_binary, read_points3D_text

def sfm_to_depth(
    sfm_dir: Path, 
    output_dir: Path,
    min_depth: float = 0.001,
    max_depth: float = 2000,
):
    # load sfm data
    try:
        cam_extrinsics = read_cameras_binary(sfm_dir / "cameras.bin")
        cam_intrinsics = read_images_binary(sfm_dir / "images.bin")
        points3d = read_points3D_binary(sfm_dir / "points3D.bin")
    except:
        cam_extrinsics = read_cameras_text(sfm_dir / "cameras.txt")
        cam_intrinsics = read_images_text(sfm_dir / "images.txt")
        points3d = read_points3D_text(sfm_dir / "points3D.txt")
    assert cam_extrinsics is not None and cam_intrinsics is not None and points3d is not None
    for img_id, img_data in tqdm(sorted(cam_intrinsics.items()), desc="sfm depths"):
        H = cam_extrinsics[img_id].height
        W = cam_extrinsics[img_id].width
        
        p3d_ids = [p3d_id for p3d_id in img_data.point3D_ids if p3d_id != -1]
        np_world_xyz = np.array([points3d[p3d_id].xyz for p3d_id in p3d_ids])
        rotation = qvec2rotmat(img_data.qvec) # quaternion to rotation matrix
        if len(p3d_ids)>0:
            points_z = (rotation @ np_world_xyz.T)[-1] + img_data.tvec[-1] # RX + T -> z
            errors = np.array([points3d[p3d_id].error for p3d_id in p3d_ids]) # reprojection error
            vis_num = np.array([len(points3d[p3d_id].image_ids) for p3d_id in p3d_ids]) # visible cameras number
            uv = np.array(
                [
                    img_data.xys[i]
                    for i in range(len(img_data.xys))
                    if img_data.point3D_ids[i] != -1
                ]
            ) # uv coordinates
            idx = np.where(
                (points_z >= min_depth)
                & (points_z <= max_depth)
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < W)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < H)
            ) # culling. you can add vis_num and errors here.
            points_z = points_z[idx]
            uv = uv[idx]
            depth_map = np.zeros((H, W), dtype=np.float32) # (H, W)
            depth_map[uv[:, 1].astype(int), uv[:, 0].astype(int)] = points_z
        else:
            depth_map = np.zeros((H, W), dtype=np.float32)
        
        img_name = Path(img_data.name).stem
        sfm_depth_path = output_dir / img_name
        np.save(sfm_depth_path, depth_map)
    
def process_mono_batch(image_list, output_dir, gpu_num, gpu_id):
    print(f"Start process in gpu {gpu_num}.")
    device = f"cuda:{gpu_id}"
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    model.eval()

    for image_path in tqdm(image_list, desc=f"GPU {gpu_num} - mono depths"):
        input_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)                       
        input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    
        prediction = model.infer(input_image)
        mono_depth = prediction["depth"]
        mono_depth_tensor = mono_depth.unsqueeze(-1)
        np_mono_depth = mono_depth_tensor.detach().cpu().numpy()
        mono_depth_path = output_dir / f"{image_path.stem}.npy"
        np.save(mono_depth_path, np_mono_depth)

def align_depths(sfm_list, mono_list, output_dir, isVisualize=False):
    sfm_depths = []
    mono_depths = []
    for sfm_path, mono_path in tqdm(zip(sfm_list, mono_list), desc=f"load tensors", total=len(sfm_list)):
        sfm_depth = np.load(sfm_path).astype(np.float32)
        sfm_depth = sfm_depth[..., np.newaxis] # (H,W) -> (H, W, 1)
        sfm_depths.append(sfm_depth)

        mono_depth = np.load(mono_path).astype(np.float32) # (H, W, 1)
        mono_depths.append(mono_depth)
    
    sfm_depths = np.stack(sfm_depths, axis=0)
    mono_depths = np.stack(mono_depths, axis=0)
    mono_masks = np.isfinite(mono_depths)  # exclude infinite and nan from moge
    mono_depths = np.where(mono_masks, mono_depths, 0.) # set infinite and nan to max depth
    valid_masks = (sfm_depths > 1e-4) & (sfm_depths < 1e4) & mono_masks # extract pixels from sfm points 

    print("Calculating scale and shift ...")
    scale, shift = compute_scale_and_shift(prediction=mono_depths, 
                                           target=sfm_depths, 
                                           mask=valid_masks)
    
    scale = scale[..., np.newaxis, np.newaxis]
    shift = shift[..., np.newaxis, np.newaxis]
    align_depths = scale * mono_depths + shift
    loss_value = np.mean((align_depths[valid_masks] - sfm_depths[valid_masks])**2)
    print(f"Average depth alignment error for batch depths is: {loss_value:2f}")

    # set infinite and nan to max depth
    align_depths = np.where(mono_masks, align_depths, 1e2)
    
    for idx, (sfm_path, aligned_depth) in enumerate(zip(sfm_list, align_depths)):
        aligned_fname = f"{Path(sfm_path).stem}_aligned.npy"
        aligned_path = output_dir / aligned_fname
        np.save(aligned_path, aligned_depth)
    
        if isVisualize and Path(sfm_path).stem == 'frame_0001': # visualize first image
            import open3d as o3d
            SCALE = 100 # for visualization

            # Reshape to 2D (1080, 1920)
            aligned_map = align_depths[idx, :, :, 0]
            sparse_map = sfm_depths[idx, :, :, 0]
            mono_mask = mono_masks[idx, :, :, 0] #mask out nan or inf points for visualization
            
            # Generate x, y coordinates
            x = np.arange(aligned_map.shape[1])
            y = np.arange(aligned_map.shape[0])
            x, y = np.meshgrid(x, y)       

            # Flatten the arrays 
            x_flat = x.flatten()
            y_flat = y.flatten()
            z_flat = aligned_map.flatten()
            mask_flat = mono_mask.flatten()
            z_flat = z_flat * SCALE
            
            # Exclude points where not mono_masks
            valid_indices = mask_flat
            x_flat = x_flat[valid_indices]
            y_flat = y_flat[valid_indices]
            z_flat = z_flat[valid_indices]
            
            # Combine into point cloud (x, y, z)
            points_aligned_map = np.vstack((x_flat, y_flat, z_flat)).T
            z_normalized = (z_flat - z_flat.min()) / (z_flat.max() - z_flat.min())
            gray_values = 1.0 - (0.8 * z_normalized)  # Map to [0.2, 1.0] for visualization
            colors_aligned_map = np.column_stack((gray_values, gray_values, gray_values))

            indices = np.nonzero(sparse_map)
            x_sparse = indices[1]
            y_sparse = indices[0]
            z_sparse = sparse_map[indices] * SCALE

            points_sparse_map = np.vstack((x_sparse, y_sparse, z_sparse)).T
            colors_sparse_map = np.tile([0, 1, 0], (points_sparse_map.shape[0], 1))

            points = np.vstack((points_aligned_map, points_sparse_map))
            colors = np.vstack((colors_aligned_map, colors_sparse_map))

            # Create an Open3D point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            # Save the point cloud as a PLY file
            debug_path = output_dir / "visualization" / "aligned_result.ply"
            debug_path.parent.mkdir(exist_ok=True)
            o3d.io.write_point_cloud(str(debug_path), pcd)
            print(f"Alignment visualized result is saved to {str(debug_path)}")

def compute_scale_and_shift(prediction, target, mask):
    '''
    *** referenced from monosdf ***
    Loss = SUM(mask * (a*prediction+b - target) ** 2)
    Return values are derived from dL/da = 0, dL/db = 0
    '''
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = np.sum(mask * prediction * prediction, axis=(1, 2))
    a_01 = np.sum(mask * prediction, axis=(1, 2))
    a_11 = np.sum(mask, axis=(1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target, axis=(1, 2))
    b_1 = np.sum(mask * target, axis=(1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = np.zeros_like(b_0)
    x_1 = np.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = np.nonzero(det)[0]

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--image_type", type=str, required=True, choices=["inpainted", "original"])
    parser.add_argument("--gpus", type=str, required=True)
    parser.add_argument("--skip_sfm", action="store_true")
    parser.add_argument("--skip_mono", action="store_true") 
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    gpu_nums = args.gpus.split(',')
    gpu_size = len(gpu_nums)
    gpu_ids = [gid for gid in range(gpu_size)]
    ctx = mp.get_context('spawn')

    # load input data
    if args.image_type == "inpainted":
        image_dir = Path(args.data_dir) / "inpainted_images"
    else:
        image_dir = Path(args.data_dir) / "images"
    assert image_dir.exists()
    image_list = sorted(list(image_dir.glob('*.png')))
    
    # load sfm data
    sfm_dir = Path(args.data_dir) / 'undistorted' / 'sparse' / '0'    
    assert sfm_dir.exists()

    # make output directories
    depth_dir = Path(args.data_dir) / 'depths'
    sfm_depth_dir = depth_dir / 'sfm_depths'
    sfm_depth_dir.mkdir(exist_ok=True, parents=True)
    mono_depth_dir = depth_dir / f'{args.image_type}_mono'
    mono_depth_dir.mkdir(exist_ok=True, parents=True)
    aligned_depth_dir = depth_dir / f'{args.image_type}_aligned'
    aligned_depth_dir.mkdir(exist_ok=True, parents=True)

    if not args.skip_sfm:
        sfm_to_depth(sfm_dir,sfm_depth_dir)
    
    if not args.skip_mono:
        batch_size = len(image_list) // gpu_size + (1 if len(image_list) % gpu_size != 0 else 0)
        with ProcessPoolExecutor(max_workers=gpu_size, mp_context=ctx) as executor:
            process_list = []
            for gpu_id in gpu_ids:
                gpu_id = int(gpu_id) % gpu_size
                if gpu_id == gpu_size - 1: # last batch
                    images_batch = image_list[gpu_id * batch_size:]
                else: # other batches
                    images_batch = image_list[gpu_id * batch_size : (gpu_id + 1) * batch_size]
                process_arg = (
                    images_batch,
                    mono_depth_dir,
                    gpu_nums[gpu_id],
                    gpu_id
                )
                process_list.append(executor.submit(process_mono_batch, *process_arg))
            for process in process_list:
                process.result() # wait for all process to complete

    # load depth datas
    sfm_list = sorted(list(sfm_depth_dir.glob('*.npy')))
    sfm_name_list = [file.stem for file in sfm_list]
    mono_list = sorted([
        file for file in mono_depth_dir.glob('*.npy')
        if "_aligned.npy" not in file.stem and file.stem in sfm_name_list
    ])
    assert len(sfm_list) == len(mono_list)
    unmatch_list = [
        file for file in mono_depth_dir.glob('*.npy')
        if file.stem not in sfm_name_list
    ]
    align_depths(sfm_list, mono_list, aligned_depth_dir, args.visualize)