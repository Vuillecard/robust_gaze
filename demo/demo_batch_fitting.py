import os 
from robust_gaze.utils.object_utils import load_obj_file, fit_3d_object
from robust_gaze.utils.visualize_utils import plot_3d_vertices
from pytorch3d.io import save_obj
import numpy as np
import torch

canonical_metric_landmarks_list = [3526,2736,1600,3542,1649,2012,1962,3908,3173,2252,785]

obj_file_source = '/Users/pierre/PhD/code_base/robust_gaze/data/reference_face_glasses/face_and_glasses_v2.obj'
obj_file_glasses = '/Users/pierre/PhD/code_base/robust_gaze/data/45-oculos/oculos.obj'

def main_fit(path_file_target,path_save_obj):

    # load the face and glasses object
    face_glasses_source ,_= load_obj_file(obj_file_source)

    # Split the face and glasse object
    face_source = face_glasses_source[:5023]
    glasses_source = face_glasses_source[5023:]

    # load the target face
    face_target,f_target= load_obj_file(path_file_target)

    # Compute the transformation to go from original glasses to glasses on target face
    transform_mat, scale = fit_3d_object(face_target.numpy()[canonical_metric_landmarks_list].T,face_source.numpy()[canonical_metric_landmarks_list].T)

    # load glasses object
    _,face_glasses = load_obj_file(obj_file_glasses)

    # Apply the transformation to the glasses
    vertex_glasses_out = scale*(glasses_source@transform_mat[:3,:3].T + transform_mat[:3,3])
    
    # concatenate glasses and face 
    out_vertex = np.concatenate((face_target.numpy(),vertex_glasses_out),axis=0)
    out_face = np.concatenate((f_target.verts_idx,face_glasses.verts_idx+5023),axis=0)

    save_obj(path_save_obj, torch.from_numpy(out_vertex),torch.from_numpy(out_face)  )
    

def main(args):

    for path_file_target in args.obj_file_target:
        path_target = os.path.join(path_file_target,'mesh_coarse.obj')
        path_save_obj = os.path.join(path_file_target,'mesh_coarse_with_glasses.obj')
        main_fit(path_target,path_save_obj)

        path_target = os.path.join(path_file_target,'mesh_coarse_trans.obj')
        path_save_obj = os.path.join(path_file_target,'mesh_coarse_trans_with_glasses.obj')
        main_fit(path_target,path_save_obj)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    dir_data = '/Users/pierre/Downloads/EMOCA_v2_lr_mse_20/'

    parser.add_argument('--obj_file_target', nargs='+', default=[dir_data + '00003900', 
                                                                    dir_data + '00007100',
                                                                    dir_data + '00016000',
                                                                    dir_data + '00023100',])
    
    args = parser.parse_args()
    main(args)