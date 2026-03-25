import pickle
import numpy as np
from plyfile import PlyData, PlyElement
import numpy as np
import cv2
import os
import joblib
from pathlib import Path

def read_pickle(fname):
    try:
        with open(fname, 'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        print(f"[Error] {e}")

        import pickle5
        with open(fname, 'rb') as f:
            data_dict = pickle5.load(f)

    return data_dict

def write_pickle(fname, data):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

def load_ply(path):
    plydata = PlyData.read(path)
    vertex_data = plydata['vertex'].data
    xyz = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    rgb = np.vstack([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T
    return xyz, rgb

def storePly(path, xyz, rgb, normals=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def save_rgb_image(tensor, path:str = None):
    array = tensor.detach().cpu().numpy()
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = (array * 255).astype(np.uint8)
    array = np.transpose(array, (1, 2, 0))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, array)

def save_depth_image(tensor, path:str = None):
    array = tensor.detach().cpu().numpy()
    array = (array - np.min(array)) / (np.max(array) - np.min(array)) * 255
    array = array.astype(np.uint8)
    array = np.transpose(array, (1, 2, 0))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, array)


def save2images(left, right, path:str = None):
    array1 = left.detach().cpu().numpy()
    array2 = right.detach().cpu().numpy()

    array1 = np.nan_to_num(array1, nan=0.0, posinf=0.0, neginf=0.0)
    array2 = np.nan_to_num(array2, nan=0.0, posinf=0.0, neginf=0.0)
    array1 = (array1 * 255).astype(np.uint8)
    array2 = (array2 * 255).astype(np.uint8)

    array1 = np.transpose(array1, (1, 2, 0))
    array2 = np.transpose(array2, (1, 2, 0))

    array1 = cv2.cvtColor(array1, cv2.COLOR_BGR2RGB)
    array2 = cv2.cvtColor(array2, cv2.COLOR_BGR2RGB)
    
    cv2.imwrite(path, np.concatenate((array1, array2), axis=1))

def save4images(left_top, right_top, left_bottom, right_bottom, path:str = None):
    array1 = left_top.detach().cpu().numpy()
    array2 = right_top.detach().cpu().numpy()
    array3 = left_bottom.detach().cpu().numpy()
    array4 = right_bottom.detach().cpu().numpy()

    array1 = np.nan_to_num(array1, nan=0.0, posinf=0.0, neginf=0.0)
    array2 = np.nan_to_num(array2, nan=0.0, posinf=0.0, neginf=0.0)
    array3 = np.nan_to_num(array3, nan=0.0, posinf=0.0, neginf=0.0)
    array4 = np.nan_to_num(array4, nan=0.0, posinf=0.0, neginf=0.0)
    
    array1 = (array1 * 255).astype(np.uint8)
    array2 = (array2 * 255).astype(np.uint8)
    array3 = (array3 * 255).astype(np.uint8)
    array4 = (array4 * 255).astype(np.uint8)

    array1 = np.transpose(array1, (1, 2, 0))
    array2 = np.transpose(array2, (1, 2, 0))
    array3 = np.transpose(array3, (1, 2, 0))
    array4 = np.transpose(array4, (1, 2, 0))

    array1 = cv2.cvtColor(array1, cv2.COLOR_BGR2RGB)
    array2 = cv2.cvtColor(array2, cv2.COLOR_BGR2RGB) 
    array3 = cv2.cvtColor(array3, cv2.COLOR_BGR2RGB)
    array4 = cv2.cvtColor(array4, cv2.COLOR_BGR2RGB)

    top_row = np.concatenate((array1, array2), axis=1)
    bottom_row = np.concatenate((array3, array4), axis=1)
    full_image = np.concatenate((top_row, bottom_row), axis=0)

    cv2.imwrite(path, full_image)
    
def save_video(path, video_name, fps = 30, format="png"):
    cmd = f'ffmpeg -y -f image2 -framerate {str(int(fps))} -i {str(path)}/%5d.{format} -c:v libx264 -pix_fmt yuv420p {str(path)}/../{video_name} -loglevel quiet' #
    os.system(cmd)
    
    cmd = f"rm -r {path}"
    os.system(cmd)

