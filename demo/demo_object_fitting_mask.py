import os 
from robust_gaze.utils.object_utils import load_obj_file, fit_3d_object
from robust_gaze.utils.visualize_utils import plot_3d_vertices
from pytorch3d.io import save_obj
import numpy as np
import torch

canonical_metric_landmarks_list = [3526,2736,1600,3542,1649,2012,1962,3908,3173,2252,785]

def main(args):

    # load the face and glasses object
    face_mask_source ,face_face= load_obj_file(args.obj_file_source)

    # Split the face and glasse object
    face_source = face_mask_source[:5023]
    mask_source = face_mask_source[5023:]

    # load the target face
    face_target,f_target= load_obj_file(args.obj_file_target)

    # Compute the transformation to go from original glasses to glasses on target face
    transform_mat, scale = fit_3d_object(face_target.numpy()[canonical_metric_landmarks_list].T,face_source.numpy()[canonical_metric_landmarks_list].T)

    # load glasses object
    _,face_glasses = load_obj_file(args.obj_file_glasses)

    # Apply the transformation to the glasses
    mask_glasses_out = scale*(mask_source@transform_mat[:3,:3].T + transform_mat[:3,3])
   
    # concatenate glasses and face 
    out_vertex = np.concatenate((face_target.numpy(),mask_glasses_out),axis=0)
    out_face = np.concatenate((f_target.verts_idx,face_face.verts_idx[5023:]+5023),axis=0)
    
    save_obj(args.save_obj, torch.from_numpy(out_vertex),torch.from_numpy(out_face)  )
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_file_source', type=str, default='/Users/pierre/PhD/code_base/robust_gaze/data/reference_face_glasses/face_with_mask.obj')
    parser.add_argument('--obj_file_target', type=str, default='/Users/pierre/Downloads/v1_mesh_coarse.obj')
    parser.add_argument('--obj_file_glasses', type=str, default='/Users/pierre/PhD/code_base/robust_gaze/data/45-oculos/oculos.obj')

    #parser.add_argument('--save_file', type=str, default='/Users/pierre/PhD/code_base/robust_gaze/output/glasses_fitted.obj')
    parser.add_argument('--save_obj', type=str, default='/Users/pierre/PhD/code_base/robust_gaze/output/mesh_coarse_with_mask.obj')
    args = parser.parse_args()
    main(args)


