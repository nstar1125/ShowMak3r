
import os
import random
import torch
import cv2
import numpy as np
from tqdm import tqdm

from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    PointLights,
    AlphaCompositor,
    HardPhongShader,
)
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer.mesh import Textures

from showmak3r.utils.camera_utils import fov2focal
from showmak3r.utils.graphics_utils import project_points_to_cam
from showmak3r.utils.image_utils import img_add_text
from showmak3r.utils.graphics_utils import get_color
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch3d")

def camdict_to_torch3d(camdict, device, zoom_scale=1.):
    img_size = [int(camdict['H']), int(camdict['W'])]

    if 'f' in camdict:
        fx = camdict['f'] * zoom_scale
        fy = camdict['f'] * zoom_scale
    else:
        fx = camdict['fx'] * zoom_scale
        fy = camdict['fy'] * zoom_scale

    cx = camdict['cx']
    cy = camdict['cy']

    focal_length = torch.tensor([fx, fy]).unsqueeze(0).to(device).float()
    principal_point = torch.tensor([cx, cy]).unsqueeze(0).to(device).float()

    cam_R = torch.diag(torch.tensor([1, 1, 1]))[None].float()
    cam_T = torch.zeros(3)[None].float() 
    cam_R[:, :2, :] *= -1.0
    cam_T[:, :1] *= -1.0
    cam_T[:, :2] *= -1.0
    
    cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point, R=cam_R, T=cam_T, device=device, in_ndc=False, image_size=[img_size])

    return cameras

def render_smpl(
    camdicts,
    shot_dicts,
    smpl_model,
    device
):
    '''
    Render SMPLs in shot_dicts from camdicts using pytorch3d

    Args:
        camdicts, shot_dicts, smpl_model, device

    Returns:
        result_imgs: dict, rendered images for each frame
        smpl_alphas: dict, alpha values for each frame
    '''
    
    # flatten shot_dicts -> people_dicts
    people_dicts = dict()
    for shot_id, shot_dict in shot_dicts.items():
        for pnum, person_dict in shot_dict.items():
            people_dicts[pnum] = person_dict
    
    # rasterization settings
    first_camera = list(camdicts.values())[0]
    img_size = [first_camera['H'], first_camera['W']]
    smpl_faces = smpl_model.smpl.faces
    smpl_faces = torch.tensor(smpl_faces.astype(np.int64)).to(device)
    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        # bin_size=64,
    )

    # define smpl colors
    smpl_colors = dict()
    _, color_lists = get_color(idx=0, interval=1, get_color_lists=True, theme=['green', 'amber', 'indigo'])
    while (len(color_lists) < len(people_dicts)): # if more then 42 people, extend color list 
        color_lists = color_lists + color_lists
    indices = random.sample(range(len(color_lists)), len(people_dicts)) # color -> pnum mapping
    for i, pnum in zip(indices, people_dicts.keys()):
        smpl_colors[pnum] = torch.tensor(color_lists[i], dtype=torch.float32).to(device) / 255.
    
    smpl_alphas = dict()
    result_imgs = dict()
    # start rendering
    for idx, fname in tqdm(enumerate(sorted(list(camdicts.keys()))), desc="Rendering", total=len(camdicts)):
        # define renderer
        camdict = camdicts[fname]
        torch3d_camera = camdict_to_torch3d(camdict, device)
        mesh_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=torch3d_camera, raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=torch3d_camera)  # Use the default shader
        )

        # set SMPL vertices
        smpl_verts_dict = dict()
        for pnum, person_dict in people_dicts.items():
            if fname not in person_dict:
                continue
            if person_dict[fname]['smpl_param'] is None:
                continue
            
            smpl_param = person_dict[fname]['smpl_param'].squeeze().float()
            smpl_param = smpl_param.unsqueeze(0).to(device)
            smpl_output = smpl_model(smpl_param)
            smpl_verts = smpl_output['smpl_verts'].detach()
            smpl_verts_dict[pnum] = smpl_verts.squeeze()
        
        w2c = torch.from_numpy(camdict['w2c']).to(device).float()
        if len(smpl_verts_dict) == 0:
            smpl_image = torch.zeros(img_size).float().cuda()[None].unsqueeze(-1).repeat(1,1,1,4)
        else:
            render_faces = []
            render_verts = []
            render_rgbs = []    

            for pnum, smpl_verts in smpl_verts_dict.items():
                smpl_verts_homo = torch.cat([smpl_verts, torch.ones_like(smpl_verts[...,0:1])], axis=-1)
                smpl_verts_pj = torch.einsum('ij,bj->bi', w2c, smpl_verts_homo) # projection
                smpl_verts_pj = smpl_verts_pj[...,:3] / (smpl_verts_pj[..., 3:] + 1e-9) # normalize
                
                render_faces.append(smpl_faces)
                render_verts.append(smpl_verts_pj)
                render_rgbs.append(smpl_colors[pnum].reshape(1, -1).repeat(len(smpl_verts_pj), 1))

            mesh = Meshes(verts=render_verts, faces=render_faces, textures=Textures(verts_rgb=render_rgbs))
            mesh = join_meshes_as_scene(mesh, True)
            smpl_image = mesh_renderer(meshes_world=mesh)
        
        smpl_alphas[fname] = (smpl_image[0][...,3:] * 255).squeeze().detach().cpu().numpy().astype(np.uint8)
        
        final_image = smpl_image[0][...,:3] * smpl_image[0][...,3:] + (1-smpl_image[0][...,3:]) * torch.ones_like(smpl_image[0][...,:3])
        final_image = final_image.detach().cpu().squeeze().numpy()
        final_image[final_image>1] = 1
        final_image = (final_image * 255).astype(np.uint8)
        final_image[...,[2,1,0]]
        result_imgs[fname] = final_image
    
    return result_imgs, smpl_alphas