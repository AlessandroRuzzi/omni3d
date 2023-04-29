from inspect import isdatadescriptor
import multiprocessing
from random import choices
import h5py
import os
import argparse
import numpy as np
import trimesh
from scipy.interpolate import RegularGridInterpolator
import time
import pdb
from termcolor import cprint
import warnings
import json
from os.path import join
from pathlib import Path
import torch
from pytorch3d.io import load_ply, save_obj
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
import sys
sys.path.append("/local/home/aruzzi/interaction-learning/AutoSDF-code")
from dataset.intercap_utils import *


warnings.filterwarnings("ignore")

def get_sdf(sdf_file, sdf_res):
    intsize = 4
    floatsize = 8

    sdf = {
        "param": [],
        "value": []
    }
    with open(sdf_file, "rb") as f:
        try:
            bytes = f.read()
            ress = np.fromstring(bytes[:intsize * 3], dtype=np.int32)
            if -1 * ress[0] != sdf_res or ress[1] != sdf_res or ress[2] != sdf_res:
                raise Exception(sdf_file, "sdf_res not consistent with ", str(sdf_res))
            positions = np.fromstring(bytes[intsize * 3:intsize * 3 + floatsize * 6], dtype=np.float64)
            sdf["param"] = [positions[0], positions[1], positions[2],
                            positions[3], positions[4], positions[5]]
            sdf["param"] = np.float32(sdf["param"])
            sdf["value"] = np.fromstring(bytes[intsize * 3 + floatsize * 6:], dtype=np.float32)
            sdf["value"] = np.reshape(sdf["value"], (sdf_res + 1, sdf_res + 1, sdf_res + 1)) # somehow the cube is sdf_res+1 rather than sdf_res... need to investigate why
        finally:
            f.close()
    return sdf

def create_h5_sdf_pt(h5_file, sdf_file, norm_obj_file, centroid, m, sdf_res):
    sdf_dict = get_sdf(sdf_file, sdf_res)

    norm_params = np.concatenate((centroid, np.asarray([m]).astype(np.float32)))
    f1 = h5py.File(h5_file, 'w')
    f1.create_dataset('pc_sdf_sample', data=sdf_dict["value"].astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('norm_params', data=norm_params, compression='gzip', compression_opts=4)
    f1.create_dataset('sdf_params', data=sdf_dict["param"], compression='gzip', compression_opts=4)
    f1.close()

    command_str = f"rm -rf {norm_obj_file}"
    os.system(command_str)
    command_str = f"rm -rf {sdf_file}"
    os.system(command_str)

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.
    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = trimesh.Trimesh(vertices=scene_or_mesh.vertices, faces=scene_or_mesh.faces)
    return mesh

# from DISN create_point_sdf_grid
def get_normalize_mesh(model_file, norm_mesh_sub_dir):
    total = 16384
    # print("[*] trimesh_load:", model_file)
    mesh_list = trimesh.load_mesh(model_file, process=False)

    mesh = as_mesh(mesh_list) # from s2s
    if not isinstance(mesh, list):
        mesh_list = [mesh]

    area_sum = 0
    area_lst = []
    for idx, mesh in enumerate(mesh_list):
        area = np.sum(mesh.area_faces)
        area_lst.append(area)
        area_sum+=area
    area_lst = np.asarray(area_lst)
    amount_lst = (area_lst * total / area_sum).astype(np.int32)
    points_all=np.zeros((0,3), dtype=np.float32)
    for i in range(amount_lst.shape[0]):
        mesh = mesh_list[i]
        points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
        points_all = np.concatenate([points_all,points], axis=0)
    centroid = np.mean(points_all, axis=0)
    points_all = points_all - centroid
    m = np.max(np.sqrt(np.sum(points_all ** 2, axis=1)))
    obj_file = os.path.join(norm_mesh_sub_dir, "pc_norm.obj")
    ori_mesh_list = trimesh.load_mesh(model_file, process=False)
    ori_mesh = as_mesh(ori_mesh_list)
    ori_mesh.vertices = (ori_mesh.vertices - centroid) / float(m)
    ori_mesh.export(obj_file)

    # print("[*] export_mesh: ", obj_file)
    return obj_file, centroid, m


def create_one_sdf(sdfcommand, sdf_res, expand_rate, sdf_file, obj_file, indx):
    command_str = f"{sdfcommand} {obj_file} {sdf_res} {sdf_res} {sdf_res} -s -e {expand_rate}"
    command_str = f"{command_str} -o {indx}.dist -m 1 -c > /dev/null"
    os.system(command_str)
    command_str2 = f"mv {indx}.dist {sdf_file}"
    os.system(command_str2)

def create_sdf_obj(sdfcommand, norm_mesh_dir, sdf_dir, obj,
       sdf_res, expand_rate, indx, h5_file=None, is_ply=False):
    suffix = ".ply" if is_ply else ".obj"

    model_id = obj.split("sequences")[-1].replace(suffix, '').replace("/","_").replace(' ', '_')

    norm_mesh_sub_dir = os.path.join(norm_mesh_dir, model_id)
    sdf_sub_dir = os.path.join(sdf_dir, model_id)

    if not os.path.exists(norm_mesh_sub_dir): os.makedirs(norm_mesh_sub_dir)
    if not os.path.exists(sdf_sub_dir): os.makedirs(sdf_sub_dir)

    sdf_file = os.path.join(sdf_sub_dir, "isosurf.sdf")
    if h5_file is None:
        h5_file = obj.replace(suffix, '_sdf.h5')

    model_file = obj
    norm_obj_file, centroid, m = get_normalize_mesh(model_file, norm_mesh_sub_dir)

    create_one_sdf(sdfcommand, sdf_res, expand_rate, sdf_file, norm_obj_file, indx)
    create_h5_sdf_pt(h5_file, sdf_file, norm_obj_file, centroid, m, sdf_res)

def process_one_obj(sdfcommand,
                    LIB_command,
                    sdf_res,
                    expand_rate,
                    obj_file,
                    indx=0,
                    is_ply=False):
    '''
    Usage: SDFGen <filename> <dx> <padding>
    Where:
        res is number of grids on xyz dimension
        w is narrowband width
        expand_rate is sdf range of max x,y,z
    '''
    cprint(f"started calling process_one_obj on {obj_file}", "red")
    os.system(LIB_command)

    tmp_dir = f'tmp/for_sdf'
    norm_mesh_dir = f'{tmp_dir}/norm_mesh'
    sdf_dir = f'{tmp_dir}/sdf'

    if not os.path.exists(norm_mesh_dir): os.makedirs(norm_mesh_dir)
    if not os.path.exists(sdf_dir): os.makedirs(sdf_dir)

    create_sdf_obj(sdfcommand, norm_mesh_dir, sdf_dir, obj_file, sdf_res,
            expand_rate, indx, is_ply=is_ply)


    cprint("[*] finished!", "green")

def load_mesh_with_texture(path):
    verts, faces = load_ply(path)
    verts_rgb = 0.5 * torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb)
    return verts, faces, textures


def get_rotated_mesh_path(mesh_path, k):
    path = Path(mesh_path)
    root = path.parent
    mesh_name = path.stem
    return str(root / f'{mesh_name}_k{k}.obj')


def generate_rotated_meshes(camera_root, mesh_path):
    seg = get_seg(mesh_path)
    verts, faces, _ = load_mesh_with_texture(mesh_path)
    results = []
    for frame in range(6):
        r, t = get_rotation_translation(camera_root, seg, frame)
        new_verts = (r@verts.T ).T + t
        obj_path = get_rotated_mesh_path(mesh_path, frame)
        save_obj(f=obj_path, verts=new_verts, faces=faces)
        results.append(obj_path)
    return results


# Pool class that gets a functions, adds it to its pool and runs it async if a worker is free
class Pool:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.pool = multiprocessing.Pool(num_workers)
        self.tasks = []

    def add_task(self, func, *args, **kwargs):
        self.tasks.append(self.pool.apply_async(func, args, kwargs))

    def wait_completion(self):
        for task in self.tasks:
            task.get()
        self.pool.close()
        self.pool.join()


if __name__ == "__main__":
    pool = Pool(multiprocessing.cpu_count())
    task_num = 0
    # TODO: set the path for the dataset (https://intercap.is.tue.mpg.de/download.php) and the path for camera related data (https://github.com/YinghaoHuang91/InterCap/tree/master/Data)
    DATASET_PATH = "/data/aruzzi/InterCap"
    CALIB_PATH = "/data/aruzzi/InterCap/Data"

    for root, dirs, files in os.walk(DATASET_PATH):
        if "Mesh" not in root:
            continue
        for ply_file in files:
            if "second_obj" not in ply_file or ".ply" not in ply_file:
                continue
            ply_path = join(root, ply_file)
            obj_paths = generate_rotated_meshes(CALIB_PATH, ply_path)

            sdfcommand = '/local/home/aruzzi/interaction-learning/AutoSDF-code/isosurface/computeDistanceField'
            lib_cmd = '/local/home/aruzzi/interaction-learning/AutoSDF-code/isosurface/LIB_PATH'

            sdf_res = 63
            expand_rate = 1.3

            for obj_file in obj_paths:
                pool.add_task(process_one_obj, sdfcommand, f"source {lib_cmd}",
                    sdf_res, expand_rate, obj_file, task_num, False)
                task_num+=1
                break
            break
        break
    pool.wait_completion()