import os 
from robust_gaze.utils.object_utils import load_obj_file, fit_3d_object
from robust_gaze.utils.visualize_utils import plot_3d_vertices
from pytorch3d.io import save_obj, load_objs_as_meshes
from pytorch3d.io import IO 
from pytorch3d.structures import Meshes
import numpy as np
import torch
import shutil
import argparse

# Index of the landmarks used for the fitting between 3d face model
FIT_SUPPORT_INDEX = [3526,2736,1600,3542,1649,2012,1962,3908,3173,2252,785]
dir_name = os.path.dirname(__file__)
dir_3d_object = os.path.join(dir_name,'object_list')

OBJ_3D = { 'glasses' : os.path.join(dir_3d_object,'glasses','glasses_template.obj'),
           'cap' : os.path.join(dir_3d_object,'cap','cap_template.obj'),
           'mask': os.path.join(dir_3d_object,'mask','mask_template.obj')}

def main(args):

    torch3d_manager = IO()
    device = 'cpu'

    # load the template face 
    path_head_template = os.path.join(dir_3d_object,'head_template','head_template.obj')
    face_source ,_= load_obj_file(path_head_template)

    # load the target face
    face_target,_= load_obj_file(args.obj_file_target)

    # Compute the transformation to go from original glasses to glasses on target face
    transform_mat, scale = fit_3d_object(face_target.numpy()[FIT_SUPPORT_INDEX].T,face_source.numpy()[FIT_SUPPORT_INDEX].T)

    # load object
    mesh_object = load_objs_as_meshes([OBJ_3D[args.obj_name]], device=device,load_textures=True)
    object_verts = mesh_object.verts_list()[0]
    object_faces = mesh_object.faces_list()[0]

    # Apply the transformation to the glasses
    vertex_glasses_out = scale*(object_verts@transform_mat[:3,:3].T + transform_mat[:3,3])

    # save the fitted object
    new_mesh = Meshes(verts=[vertex_glasses_out], faces=[object_faces], textures=mesh_object.textures)
    torch3d_manager.save_mesh(new_mesh,os.path.join(args.save_obj,args.obj_name+'.obj'))
    

if __name__ == "__main__":

    person_ids = ['sample%d'%i for i in range(1,6)]
    define_path = lambda x,file: os.path.join('/Users/pierre/Downloads/EVE/EMOCA_v2_lr_mse_20/'+x+'/'+file)

    for person_id in person_ids:
        print(person_id)
        parser = argparse.ArgumentParser()
        parser.add_argument('--obj_file_target', type=str, default=define_path(person_id,'mesh_coarse_trans.obj'))
        parser.add_argument('--obj_name', type=str, default='mask', choices=['glasses','cap','mask'])
        parser.add_argument('--save_obj', type=str, default=define_path(person_id,''))

        args = parser.parse_args()

        for object in ['glasses','cap','mask']:
            print(object)
            args.obj_name = object
            main(args)

