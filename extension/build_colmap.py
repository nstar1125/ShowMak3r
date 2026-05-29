import numpy as np
import pycolmap
import copy

def res_to_pycolmap(
    points3d,
    points_xyf,
    points_rgb,
    extrinsics,
    intrinsics,
    image_size,
    shared_camera=False,
    camera_type="PINHOLE",
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP
    It saves points3d to colmap reconstruction format to serve as init for Gaussians or other nvs methods.

    Args:
        points3d: (P, 3)
        points_xyf: (P, 3), with x, y coordinates and frame indices
        points_rgb: (P, 3), rgb colors
        extrinsics: (N, 4, 4)
        intrinsics: (N, 3, 3)
        image_size: (2,), assume all the frames have been padded to the same size
    """
    N = len(extrinsics)
    P = len(points3d)

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    for vidx in range(P):
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx])

    camera = None
    # frame idx
    for fidx in range(N):
        # set camera
        if camera is None or (not shared_camera):
            pycolmap_intri = np.array([intrinsics[fidx][0, 0], 
                                       intrinsics[fidx][1, 1], 
                                       intrinsics[fidx][0, 2], 
                                       intrinsics[fidx][1, 2]]) # (fx, fy, cx, cy)

            camera = pycolmap.Camera(
                model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, 
                camera_id=fidx + 1
            )

            # add camera
            reconstruction.add_camera(camera)

        # set image
        c2w = extrinsics[fidx]
        w2c = np.linalg.inv(c2w)
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(w2c[:3, :3]), w2c[:3, 3]
        )  # Rot and Trans

        image = pycolmap.Image(
            id=fidx + 1, name=f"image_{fidx + 1}", 
            camera_id=camera.camera_id, cam_from_world=cam_from_world
        )

        points2D_list = []

        point2D_idx = 0

        points_belong_to_fidx = points_xyf[:, 2].astype(np.int32) == fidx
        points_belong_to_fidx = np.nonzero(points_belong_to_fidx)[0]

        for point3D_batch_idx in points_belong_to_fidx:
            point3D_id = point3D_batch_idx + 1
            point2D_xyf = points_xyf[point3D_batch_idx]
            point2D_xy = point2D_xyf[:2]
            points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

            # add element
            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            image.registered = True
        except:
            print(f"frame {fidx + 1} does not have any points")
            image.registered = False

        # add image
        reconstruction.add_image(image)

    return reconstruction

def rename_and_rescale(
    reconstruction, 
    image_paths, 
    original_coords, 
    img_size, 
    shift_point2d_to_original_res=False, 
    shared_camera=False
):
    """
    Rename and rescale the camera parameters and point2D to the original size
    """
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1].name

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratios = real_image_size / img_size

            pred_params = pred_params * np.tile(resize_ratios, 2)
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = int(real_image_size[0])
            pycamera.height = int(real_image_size[1])

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratios

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction