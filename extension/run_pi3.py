import torch
import torch.nn.functional as F
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import utils3d
from moge.utils.geometry_torch import recover_focal_shift

sys.path.append("./submodules/Pi3")
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3 

from extension.build_colmap import res_to_pycolmap, rename_and_rescale

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str,
                        help="Path to the data directory.")
    parser.add_argument("--mask_path", type=str,
                        help="Path to the input mask directory.")
    parser.add_argument("--save_path", type=str,
                        help="Path to save the output files.")
    parser.add_argument("--num_points", type=int, default=10_000,
                        help="Number of points to sample. Default: 10_000")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    return parser.parse_args()

def main(args):
    image_paths = sorted(list(Path(args.data_path).glob("*.png")))
    mask_paths = sorted(list(Path(args.mask_path).glob("*.png.png")))
    # we assume all images have the same size
    first_img = cv2.imread(image_paths[0])
    original_height, original_width = first_img.shape[:2]
    original_coords = np.stack([np.array([0., 0., original_width, original_height])] * len(image_paths), axis=0)

    print(f'Sampling point number: {args.num_points}')

    # 1. Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`

    # 2. Prepare input data
    # The load_images_as_tensor function will print the loading path
    imgs = load_images_as_tensor(args.data_path, interval=1).to(device) # (N, 3, H, W)

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs[None]) # Add batch dimension

    # 4. process data
    points3d = res["points"][0] # (N, H, W, 3)
    
    conf_masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.05) # 0.03
    masks = torch.logical_and(conf_masks, non_edge)[0]

    # append dynamic masks
    dynamic_mask_list = []
    for mask_path in mask_paths:
        dynamic_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        dynamic_mask = torch.from_numpy(dynamic_mask).to(device)
        dynamic_mask = dynamic_mask / 255.0
        # reverse
        dynamic_mask = 1 - dynamic_mask
        # increase dynamic size
        kernel_size = int(max(original_height, original_width) * 0.02)
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
        dynamic_mask = F.conv2d(dynamic_mask[None, None].float(), kernel, padding=kernel_size//2)[0, 0]
        # resize
        dynamic_mask = F.interpolate(dynamic_mask[None, None].float(), 
                                     size=imgs.shape[-2:], 
                                     mode='bilinear', 
                                     align_corners=False)[0, 0] # (H, W)
        # reverse back
        dynamic_mask = ~dynamic_mask.bool()
        dynamic_mask_list.append(dynamic_mask)
    dynamic_masks = torch.stack(dynamic_mask_list, dim=0)
    masks = torch.logical_and(masks, dynamic_masks)
    
    # sample random points per frame to reduce memory
    rand_masks = torch.zeros_like(masks, dtype=torch.bool)
    for frame_idx in range(masks.shape[0]):
        idx = torch.randperm(masks.shape[1] * masks.shape[2], device=device)[:args.num_points]
        rand_masks.view(masks.shape[0], -1)[frame_idx, idx] = True
    masks = torch.logical_and(masks, rand_masks)
    
    sampled_points3d = points3d[masks] # (P, 3)
    sampled_imgs = imgs.permute(0, 2, 3, 1)[masks] # (P, 3)
    sampled_rgbs = (sampled_imgs * 255).to(torch.uint8) # (P, 3)
    points_xyf_list = []
    # Create points_xyf (P, 3) with x, y coordinates and frame indices
    for frame_idx in range(0, masks.shape[0]):
        points_map = masks[frame_idx]
        y_coords, x_coords = torch.where(points_map)
        frame_indices = torch.full_like(x_coords, frame_idx, dtype=torch.float32)
        frame_xyf = torch.stack([x_coords.float(), y_coords.float(), frame_indices], dim=1)
        points_xyf_list.append(frame_xyf)
    sampled_points_xyf = torch.cat(points_xyf_list, dim=0) # (P, 3)

    # Use recover_focal_shift function from MoGe
    local_points = res["local_points"] # (N, H, W, 3)
    focal, shift = recover_focal_shift(local_points, conf_masks, downsample_size=(64, 64))
        
    # Calculate fx, fy from focal
    resize_height, resize_width = local_points.shape[-3:-1]
    aspect_ratio = resize_width / resize_height

    fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio * resize_width
    fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 * resize_height
    
    cx = resize_width//2 
    cy = resize_height//2 
    
    intr = utils3d.torch.intrinsics_from_focal_center(fx, fy, cx, cy)
    res["intrinsics"] = intr # (1, N, 3, 3)
    res["shift"] = shift # (1, N)
    
    _extrinsics = res["camera_poses"][0] # (N, 4, 4)
    _intrinsics = res["intrinsics"][0] # (N, 3, 3)

    # 5. Save ply
    print(f"Saving point cloud to: {args.save_path}/point_cloud.ply")
    write_ply(sampled_points3d.cpu(), # (P, 3)
              sampled_rgbs, # (P, 3)
              Path(args.save_path) / "point_cloud.ply")
    
    # 6. Save COLMAP data
    # save colmap data
    reconstruction = res_to_pycolmap(
        sampled_points3d.cpu().numpy(), # (P, 3)
        sampled_points_xyf.cpu().numpy(), # (P, 3)
        sampled_rgbs.cpu().numpy(), # (P, 3)
        _extrinsics.cpu().numpy(), # (N, 4, 4)
        _intrinsics.cpu().numpy(), # (N, 3, 3)
        (resize_width, resize_height),
        shared_camera=False
    )
    
    # resize to original resolution
    reconstruction = rename_and_rescale(
        reconstruction, 
        image_paths,
        original_coords,
        (resize_width, resize_height),
        shift_point2d_to_original_res=True, 
        shared_camera=False
    )
    
    print(f"Saving reconstruction to {args.save_path}/undistorted/sparse/0")
    sparse_reconstruction_dir = Path(args.save_path)/"undistorted/sparse/0"
    sparse_reconstruction_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(sparse_reconstruction_dir))

if __name__ == '__main__':
    args = parse_args()
    main(args)
    