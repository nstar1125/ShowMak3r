import torch
import tyro
import open3d as o3d
import numpy as np
import time
import cv2
import viser
import viser.transforms as tf
import matplotlib
import random
from pathlib import Path
from showmak3r.utils.camera_utils import focal2fov
from showmak3r.utils.graphics_utils import get_color
from scipy.spatial.transform import Rotation as R

CAM_SCALE = 0.05 # 0.05
LOCK_VIEW = False

def visualize_init_scene(
    people_dicts, 
    smpl_model, 
    scene_pcd, 
    cam_dicts, 
    img_dict, 
    device,
    share: bool = True,
    type: str = "mesh",
    port: int = 8080,
):
    num_frames = len(img_dict)
    ######################################################################################
    ##                                  Set viser server                                ##
    ###################################################################################### 
    server = viser.ViserServer(port=port)
    if share:
        server.request_share_url()
    
    server.scene.set_up_direction('-y')
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=False,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_view_all_frames = server.gui.add_checkbox("View All Frames", False)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=1, initial_value=10
        )
    
    # gui settings
    @gui_next_frame.on_click
    def _(_):
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_):
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    @gui_playing.on_update
    def _(_):
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    def update_frame_visibility():
        for i, frame_node in enumerate(frame_nodes):
            if not gui_view_all_frames.value:
                frame_node.visible = (i  == gui_timestep.value)
            else:
                frame_node.visible = True

    prev_timestep = 0
    @gui_timestep.on_update
    def _(_):
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        prev_timestep = current_timestep
        if LOCK_VIEW:
            update_frame_camera(int(current_timestep))
        update_frame_visibility()

    @gui_view_all_frames.on_update
    def _(_):
        update_frame_visibility()
    
    def update_frame_camera(t: int):
        if LOCK_VIEW:
            pos, look, fov = camera_poses[t]
            for client in server.get_clients().values():
                with client.atomic():
                    client.camera.position = pos
                    client.camera.look_at = look
                    client.camera.fov = fov
    
    ######################################################################################
    ##                                  Upload assets                                   ##
    ######################################################################################
    scene_points = scene_pcd[0].astype(np.float32)
    scene_rgbs = (scene_pcd[1] / 255.).astype(np.float32)
    # server.scene.add_point_cloud(
    #         "/stage",
    #         points=scene_points,
    #         colors=scene_rgbs,
    #         position=(0, 0, 0),
    #         point_shape="circle",
    #         point_size=CAM_SCALE * 0.1,
    #     )
    
    frame_nodes = []
    frame_names = sorted(img_dict.keys())

    if LOCK_VIEW:
        camera_poses = []
        world_cam_R0 = None
        world_cam_T0 = None
    
    # define smpl colors
    smpl_colors = dict()
    _, color_lists = get_color(idx=0, interval=1, get_color_lists=True, theme=['green', 'amber', 'indigo'])
    while (len(color_lists) < len(people_dicts)):
        color_lists = color_lists + color_lists
    random.seed(42) # for reproducibility
    indices = random.sample(range(len(color_lists)), len(people_dicts))
    for i, pnum in zip(indices, people_dicts.keys()):
        smpl_colors[pnum] = torch.tensor(color_lists[i], dtype=torch.float32).to(device) / 255.

    # define frame colors
    cmap = matplotlib.cm.get_cmap('Spectral')
    for i, fname in enumerate(frame_names):
        # add smpl points per timestep
        frame_node = server.scene.add_frame(f"/frames/t_{i}", show_axes=False)
        frame_nodes.append(frame_node)
        
        current_color = np.array(cmap(i / num_frames)[:3]) # gradually change from green to red

        for pnum, person_dict in people_dicts.items():
            smpl_param = person_dict[fname]['smpl_param'].squeeze().float()
            smpl_param = smpl_param.unsqueeze(0).to(device)
            smpl_output = smpl_model(smpl_param)
            
            smpl_points = smpl_output['smpl_verts'].squeeze(0).detach().cpu().numpy()
            smpl_faces = smpl_output['smpl_faces'].detach().cpu().numpy()
            
            if type == "mesh":
                server.scene.add_mesh_simple(
                                name=f"/frames/t_{i}/actor_{pnum}",
                                vertices=smpl_points,
                                faces=smpl_faces,
                                flat_shading=False,
                                wireframe=False,
                                color=smpl_colors[pnum])
            elif type == "point":
                smpl_rgbs = np.tile(current_color, (smpl_points.shape[0], 1))
                server.scene.add_point_cloud(
                    name=f"/frames/t_{i}/actor_{pnum}",
                    points=smpl_points,
                    colors=smpl_rgbs,
                    point_size=0.05,
                    point_shape="circle")
            else:
                raise ValueError(f"Invalid type: {type}")
        
        # add camera per timestep
        image = img_dict[fname]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = image_rgb / 255.
        
        H, W = cam_dicts[fname]['H'], cam_dicts[fname]['W']
        fov_x = focal2fov(cam_dicts[fname]['fx'], W)
        fov_y = focal2fov(cam_dicts[fname]['fy'], H)
        
        w2c = cam_dicts[fname]['w2c']
        c2w = np.linalg.inv(w2c)
        world_cam_R = c2w[:3, :3]
        world_cam_T = c2w[:3, 3]
        
        xyzw = R.from_matrix(world_cam_R).as_quat()
        wxyz = np.roll(xyzw, shift=1)
        
        if not LOCK_VIEW: # view manually
            server.scene.add_camera_frustum(
                name=f"/frames/t_{i}/camera",
                fov=fov_x,
                aspect=W/H,
                scale=CAM_SCALE,
                image=image_rgb,
                wxyz=wxyz,
                position=world_cam_T,
                color=current_color,
            )
            server.scene.add_frame(
                f"/frames/t_{i}/camera/axes",
                axes_length=CAM_SCALE * 0.5,
                axes_radius=CAM_SCALE * 0.05
            )
        else: # view selected trajectory
            if i == 0:
                world_cam_R0 = world_cam_R
                world_cam_T0 = world_cam_T

            scale = 0.2
            interval = 60
            shift = world_cam_R0 @ np.array([np.sin(i * (2 * np.pi / interval)), np.cos(i * (2 * np.pi / interval)), 0.0], dtype=float)
            offset = world_cam_R0 @ np.array([0.0, -0.25, -1.0], dtype=float)
            cur_pos = world_cam_T0.astype(float) + shift * scale + offset
            # cur_wxyz = wxyz.astype(float)
            forward_look = world_cam_R0 @ np.array([0.0, 0.0, 1.2], dtype=float)
            cur_look = (world_cam_T0 + forward_look).astype(float)
            cur_fov  = float(fov_x)

            camera_poses.append((cur_pos, cur_look, cur_fov))

    ######################################################################################
    ##                                  Start visualization                             ##
    ######################################################################################
    def playback_loop():
        nonlocal prev_timestep
        while True:
            if gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1) % num_frames
            time.sleep(1.0 / gui_framerate.value)

    print("[INFO] Start viser visualization.")
    update_frame_camera(0)
    playback_loop()
    
if __name__ == "__main__":
    tyro.cli(visualize_init_scene)