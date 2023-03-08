from posixpath import dirname
from warnings import formatwarning
import numpy as np
import sys 
import os 
from subprocess import call
import shutil
from torch import typename
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import torch

from torch.nn.parallel.data_parallel import DataParallel
import configs.config_loader as cfg_loader
from glob import glob

import torch.distributed as dist
import pymeshlab as ml
import trimesh
import time

from mesh_to_sdf import sample_sdf_near_surface, mesh_to_voxels, mesh_to_sdf
from mesh_to_sdf.utils import get_raster_points
from numpy.core.einsumfunc import einsum_path
from numpy.lib.twodim_base import mask_indices

import trimesh
import pyrender
import numpy as np
from trimesh import points
#import igl
from skimage.measure import marching_cubes
import torch
import os
# this is mostly from https://github.com/chrischoy/3D-R2N2/blob/master/lib/voxel.py 
# though I sped up the voxel2mesh function considerably, now only surface voxels are saved
# this is only really important for very large models 

MGN_TYPE = [
    'Pants',
    'ShortPants',
    'LongCoat',
    'ShirtNoCoat',
    'TShirtNoCoat']

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh

def voxel2mesh(voxels, threshold=.3):
    cube_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
                  [1, 1, 1]]  # 8 points

    cube_faces = [[0, 1, 2], [1, 3, 2], [2, 3, 6], [3, 7, 6], [0, 2, 6], [0, 6, 4], [0, 5, 1],
                  [0, 4, 5], [6, 7, 5], [6, 5, 4], [1, 7, 3], [1, 5, 7]]  # 12 face

    cube_verts = np.array(cube_verts)
    cube_faces = np.array(cube_faces) + 1

    l, m, n = voxels.shape

    scale = 0.01
    cube_dist_scale = 1.1
    verts = []
    faces = []
    curr_vert = 0

    positions = np.where(voxels > threshold) # recieves position of all voxels
    offpositions = np.where(voxels < threshold) # recieves position of all voxels
    voxels[positions] = 1 # sets all voxels values to 1 
    voxels[offpositions] = 0 
    for i,j,k in zip(*positions):
        if np.sum(voxels[i-1:i+2,j-1:j+2,k-1:k+2])< 27 : #identifies if current voxels has an exposed face 
            verts.extend(scale * (cube_verts + cube_dist_scale * np.array([[i, j, k]])))
            faces.extend(cube_faces + curr_vert)
            curr_vert += len(cube_verts)   
    return np.array(verts), np.array(faces)


def write_obj(filename, verts, faces):
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))


def voxel2obj(filename, pred, threshold=.3):
    verts, faces = voxel2mesh(pred, threshold )
    write_obj(filename, verts, faces)


# arange shapenet_improved dataset into folders
def preprocess_shapenet(data_path, obj_name='model.obj'):
    file_list = os.listdir(data_path)
    for obj_path in file_list:
        dir_name = os.path.join(data_path, os.path.splitext(obj_path)[0])
        os.mkdir(dir_name)
        shutil.move(os.path.join(data_path, obj_path), dir_name)
        os.rename(os.path.join(dir_name, obj_path), os.path.join(dir_name, obj_name))

        print('{} moved!'.format(obj_path))

def preprocess_mgn(data_path, obj_name='model.obj'):
    file_list = os.listdir(data_path)
    for obj_dir in file_list:
        for type_name in MGN_TYPE:
            type_path = os.path.join(data_path, obj_dir, type_name + '.obj')
            if os.path.exists(type_path):
                dir_name = os.path.join(data_path, obj_dir + '_' + type_name)
                os.mkdir(dir_name)
                shutil.move(type_path, dir_name)
                os.rename(os.path.join(dir_name, type_name + '.obj'), os.path.join(dir_name, obj_name))

                print('{} moved!'.format(dir_name))

        shutil.rmtree(os.path.join(data_path, obj_dir))

def preprocess_mixamo(data_path, obj_name='model.obj'):
    dir_list = os.listdir(data_path)
    for dir_name in dir_list:
        os.rename(os.path.join(data_path, dir_name, dir_name+'.obj'), os.path.join(data_path, dir_name, obj_name))

# fix npz files
def fix_npz(path):
    dir_list = os.listdir(path)
    bad_file_list = []
    for dir_name in dir_list:
        file_list = os.listdir(os.path.join(path, dir_name))
        for file_name in file_list:
            if '.npz' in file_name:
                try:
                    np.load(os.path.join(path,dir_name,file_name), allow_pickle=True)
                except:
                    print('bad file: {}'.format(file_name))
                    bad_file_list.append(file_name)

    return bad_file_list

def fix_npz_mp(file_path):
    try:
        np.load(file_path, allow_pickle=True)
    except:
        print('bad file: {}'.format(file_path))
        os.remove(file_path)

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def add_tail(name_list, tail='_old'):
    ret_list = []
    for name in name_list:
        ret_list.append(os.path.join(os.path.dirname(name)+tail, os.path.basename(name)))

    return ret_list

def modify_npz(npz_path):
    split = np.load(npz_path)
    np.savez(npz_path, train=add_tail(split['train']), test=add_tail(split['test']), val=add_tail(split['val']))


def pc2mesh(data_dir):
    name_list = os.listdir(data_dir)

    for name in name_list:

        target_path = os.path.join(data_dir, name, 'dense_point_cloud_7_bpa.obj')
        if os.path.exists(target_path):
            print('{} exsits, skip!'.format(name))
            continue

        print('processing {}'.format(name))
        start = time.time()
        path = os.path.join(data_dir, name, 'dense_point_cloud_7_pc.off')

        ms = ml.MeshSet()
        ms.load_new_mesh(path)
        ms.load_filter_script('ndf_postprocess.mlx')
        ms.apply_filter_script()

        ms.save_current_mesh(os.path.join(data_dir, name, 'dense_point_cloud_7_bpa.obj'))

        duration = time.time() - start

        print('duration {}'.format(duration))

def preprocess_watertight(data_dir, src_dir):
    name_list = os.listdir(data_dir)

    for name in name_list:

        target_path = os.path.join(src_dir, name, 'model_wt.obj')
        if os.path.exists(target_path):
            print('{} exsits, skip!'.format(name))
            continue

        print('processing {}'.format(name))
        start = time.time()
        path = os.path.join(src_dir, name, 'model_scaled.off')

        '''
        ms = ml.MeshSet()
        ms.load_new_mesh(path)
        ms.load_filter_script('ndf_postprocess.mlx')
        ms.apply_filter_script()

        ms.save_current_mesh(os.path.join(data_dir, name, 'dense_point_cloud_7_bpa.obj'))
        '''

        voxel_resolution = 256

        mesh = trimesh.load(path)

        points = get_raster_points(voxel_resolution=voxel_resolution)

        sdf = igl.signed_distance(points, mesh.vertices, mesh.faces)[0]

        sdf = sdf.reshape([voxel_resolution]*3)

        verts, faces, norms, vals = marching_cubes(sdf, 0)

        trimesh.Trimesh(vertices=verts, faces=faces).export(target_path)

        duration = time.time() - start

        print('duration {}'.format(duration))

def pc2mesh_perobj(obj_path):
    target_path = os.path.join(os.path.dirname(obj_path), 'dense_point_cloud_7_bpa.obj')
    if os.path.exists(target_path):
        print('{} exists, skip!'.format(obj_path))
        return

    print('processing {}'.format(obj_path))
    #path = os.path.join(data_dir, name, 'dense_point_cloud_7_pc.off')

    ms = ml.MeshSet()
    ms.load_new_mesh(obj_path)
    ms.load_filter_script('ndf_postprocess.mlx')
    ms.apply_filter_script()
    ms.save_current_mesh()

def softplusplus(input_, alpha=0.2):
    return torch.log(1.0+torch.exp(input_*(1.0-alpha)))+alpha*input_-torch.log(torch.tensor(2.0))

class SoftPlusPlus(torch.nn.Module):
    def __init__(self, alpha=0.2) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, input_):
        #print('input: {}'.format(input_))
        return softplusplus(input_, self.alpha)

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(
                -np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement
            # Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class Sine(torch.nn.Module):
    def __init__(self, const=30.):
        super().__init__()
        self.const = const

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.const * input)



if __name__=='__main__':
    #modify_npz('datasets/shapenet_improved/data/split_shapenet_cars_chen_old.npz')
    #preprocess_shapenet('datasets/shapenet_improved/data/03001627')
    #preprocess_mgn('datasets/MGN/data/0')
    #preprocess_mixamo('datasets/mixamo_data/data/0')

    #pc2mesh('experiments/shapenet_ships_chen_apex_148_3000/evaluation/generation/04530566')
    #pc2mesh('experiments/MGN_occ/evaluation_test_cd_100kp/generation/0')
    #pc2mesh('experiments/shapenet_cars_chen_apex_148/evaluation/generation/02958343')
    #pc2mesh('experiments/shapenet_cars_chen_closed_300/evaluation/generation/02958343')
    #preprocess_watertight('experiments/MGN_occ_3000/evaluation/generation/0', 'datasets/MGN/data/0')
    preprocess_watertight('experiments/shapenet_cars_chen_apex_148/evaluation/generation/02958343', 'datasets/shapenet_improved/data/02958343')

    '''
    cfg = cfg_loader.get_config()

    paths = glob( cfg.data_dir + '/*/*.npz')
    print(len(paths))

    paths = sorted(paths)

    chunks = np.array_split(paths,cfg.num_chunks)
    paths = chunks[cfg.current_chunk]


    if cfg.num_cpus == -1:
        num_cpus = mp.cpu_count()
        print('cpu count: {}'.format(num_cpus))
    else:
        num_cpus = cfg.num_cpus

    def multiprocess(func):
        p = Pool(num_cpus)
        p.map(func, paths)
        p.close()
        p.join()

    multiprocess(fix_npz_mp)
    '''
    '''
    cfg = cfg_loader.get_config()

    paths = glob( 'experiments/shapenet_cars_chen_closed_3000/evaluation/generation/02958343' 
                    + '/*/dense_point_cloud_7_pc.off')
    print(len(paths))

    chunks = np.array_split(paths,cfg.num_chunks)
    paths = chunks[cfg.current_chunk]


    if cfg.num_cpus == -1:
        num_cpus = mp.cpu_count()
        print('cpu count: {}'.format(num_cpus))
    else:
        num_cpus = cfg.num_cpus

    def multiprocess(func):
        p = Pool(num_cpus)
        p.map(func, paths)
        p.close()
        p.join()

    multiprocess(pc2mesh_perobj)
    '''