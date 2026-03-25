import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from random import randint, random
from argparse import ArgumentParser

from config.stage import ModelParams, PipelineParams, OptimizationParams
from showmak3r.pipeline.renderer.gaussian_renderer import stage_render
from showmak3r.pipeline.renderer.renderer_wrapper import render_stage_full, render_depth_maps
from showmak3r.pipeline.scene import Scene, GaussianModel
from showmak3r.utils.general_utils import safe_state
from showmak3r.utils.loss_utils import l1_loss, ssim, log_l1, tv_loss
from showmak3r.utils.system_utils import searchForMaxIteration
from showmak3r.utils.io_utils import save2images

def training(args):
    # Setup dataset, optimizer, pipeline
    dataset = lp.extract(args)
    dataset.reverse_mask = args.reverse_mask
    opt = op.extract(args)
    pipe = pp.extract(args)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # Setup arguments
    testing_iterations = args.test_iterations
    saving_iterations = args.save_iterations
    checkpoint_iterations = args.checkpoint_iterations
    checkpoint = args.start_checkpoint
    debug_from = args.debug_from
    first_iter = 0
    
    # Initialize Gaussian model and scene
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, scene_type="stage", view_dir_reg=pipe.view_dir_reg)
    gaussians.training_setup(opt)

    # load from checkpoint
    output_path = Path(scene.model_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if checkpoint:
        ckpt_path = output_path / "checkpoints"
        max_iter = searchForMaxIteration(ckpt_path)
        print(f"[INFO] Loading checkpoint from iteration {max_iter}")
        (model_params, first_iter) = torch.load(ckpt_path / f"iteration_{max_iter}.pth")
        gaussians.restore(model_params, opt)

    # Initialize background
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    white_bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    black_bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    if dataset.random_background:
        print("[INFO] Using random background")

    # Use Wandb for logging
    if pipe.use_wandb:
        import wandb
        wandb.init(project=pipe.wandb_project, name=pipe.wandb_name, resume=pipe.wandb_resume)

    # ------------------------------ Start training -----------------------------
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")    
    viewpoint_stack = None
    ema_loss_for_log = 0.0 
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            
            # select only one random video frame
            excluded_cams = []
            if iteration < opt.exclude_frames_until_iter:
                for cam in viewpoint_stack.copy():
                    if cam.fname.startswith("frame"):
                        excluded_cams.append(cam)
                        viewpoint_stack.remove(cam)
                if excluded_cams:
                    random_cam = excluded_cams.pop(randint(0, len(excluded_cams)-1))
                    viewpoint_stack.append(random_cam)
        
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        if dataset.random_background:
            invert_bg_color = random() > 0.5
            background = white_bg if invert_bg_color else black_bg
            
        render_pkg = stage_render(viewpoint_cam, gaussians, pipe, background)
        render_image, viewspace_point_tensor, visibility_filter, radii, depth, alpha = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"], render_pkg["alpha"]
        render_image[render_image > 1] /= render_image[render_image > 1] # (turn off saturated points)
        
        # Loss
        gt_image = viewpoint_cam.gt_image.cuda()
        gt_depth = viewpoint_cam.pseudo_gt_depth.unsqueeze(0).cuda()

        # Debug
        if iteration % 100 == 0:
            save2images(render_image, gt_image, f"debug/iteration_{iteration}.png")

        # filter out foreground
        if not (viewpoint_cam.gt_mask is None):
            mask = viewpoint_cam.gt_mask.cuda()
            mask = mask[:3, :, :]

            if dataset.reverse_mask:
                mask = 1 - mask
            
            render_image = render_image * mask
            gt_image = gt_image * mask

        # depth tolerance
        vis_depth = gt_depth > opt.depth_tolerance
        
        render_image = render_image*vis_depth
        gt_image = gt_image*vis_depth
        depth = depth*vis_depth
        gt_depth = gt_depth*vis_depth
            
        # rgb loss
        Ll1 = l1_loss(render_image, gt_image)
        photo_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(render_image, gt_image))

        # depth loss
        depth_loss=0
        mono_depth_loss = log_l1(depth, gt_depth.float())
        depth_loss += opt.mono_depth_lambda * mono_depth_loss
        
        # TV loss
        smooth_loss = tv_loss(depth.permute(1, 2, 0))
        depth_loss += opt.smooth_loss_lambda * smooth_loss
        
        # compute total loss
        loss = photo_loss + depth_loss
        
        if iteration > opt.end_depth:
            opt.mono_depth_lambda = 0.02
            opt.smooth_loss_lambda = 0.05

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "N_gaussian": f"{gaussians.get_n_points}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save gaussians and depth maps
            background = white_bg if dataset.white_background else black_bg
            if (iteration in saving_iterations):
                print(f"[INFO] Rendering results at iteration {iteration}")
                render_stage_full(scene, gaussians, pipe, background, iteration, output_path)    
                
                ply_path = output_path / "point_cloud"
                ply_path.mkdir(parents=True, exist_ok=True)
                gaussians.save_ply(ply_path / f"iteration_{iteration}.ply")
                print(f"[INFO] Saved Gaussians at iteration {iteration}")

                pcd_path = output_path / "simple_pcd"
                pcd_path.mkdir(parents=True, exist_ok=True)
                gaussians.save_pcd(pcd_path / f"iteration_{iteration}.ply")
                print(f"[INFO] Saved simple point cloud at iteration {iteration}")

                depth_path = output_path / "depth_maps"
                depth_path.mkdir(parents=True, exist_ok=True)
                render_depth_maps(scene, gaussians, pipe, background, depth_path)
                print(f"[INFO] Saved depth maps at iteration {iteration}")

            # Save checkpoint
            if (iteration in checkpoint_iterations):
                print(f"[INFO] Saving Checkpoint at iteration {iteration}")
                ckpt_path = output_path / "checkpoints"
                ckpt_path.mkdir(parents=True, exist_ok=True)
                torch.save((gaussians.capture(), iteration), ckpt_path / f"iteration_{iteration}.pth")

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                delete_unseen = True if iteration % 3000 == 0 else False
                delete_large = True if iteration % 3000 == 0 else False

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(max_grad=opt.densify_grad_threshold * 0.5, 
                                                min_opacity=0.005, 
                                                extent=scene.cameras_extent, 
                                                max_screen_size=size_threshold, 
                                                delete_unseen=delete_unseen,
                                                delete_large=delete_large)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = False)
            
            # Wandb
            if pipe.use_wandb and iteration % pipe.wandb_interval == 0:
                log_dict = {
                    "iteration": iteration,
                    "total_loss": loss.item(),
                    "photo_loss": photo_loss.item(),
                    "depth_loss": depth_loss.item(),
                    "num_points": gaussians.get_xyz.shape[0],
                }
                wandb.log(log_dict, step=iteration)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 10_000, 15_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[3_000, 7_000, 10_000, 15_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--reverse_mask", action='store_true')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("[INFO] Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Main
    training(args)
    print("\n[INFO] Training complete.")