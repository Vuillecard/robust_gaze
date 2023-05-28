import os 
from robust_gaze.utils.object_utils import load_obj_file, fit_3d_object
from robust_gaze.utils.visualize_utils import plot_3d_vertices
from pytorch3d.io import save_obj
import numpy as np
import torch
import shutil
canonical_metric_landmarks_list = [3526,2736,1600,3542,1649,2012,1962,3908,3173,2252,785]

def main(args):

    # load the face and glasses object
    face_glasses_source ,_= load_obj_file(args.obj_file_source)

    # Split the face and glasse object
    face_source = face_glasses_source[:5023]
    glasses_source = face_glasses_source[5023:]

    # load the target face
    face_target,f_target= load_obj_file(args.obj_file_target)

    # Compute the transformation to go from original glasses to glasses on target face
    transform_mat, scale = fit_3d_object(face_target.numpy()[canonical_metric_landmarks_list].T,face_source.numpy()[canonical_metric_landmarks_list].T)

    # load glasses object
    _,face_glasses = load_obj_file(args.obj_file_glasses)

    # Apply the transformation to the glasses
    vertex_glasses_out = scale*(glasses_source@transform_mat[:3,:3].T + transform_mat[:3,3])
    
    # concatenate glasses and face 
    # out_vertex = np.concatenate( (face_target.numpy(),vertex_glasses_out), axis=0 )
    # out_face = np.concatenate( (f_target.verts_idx,face_glasses.verts_idx+5023), axis=0 )

    # save_obj( args.save_obj, torch.from_numpy(out_vertex),torch.from_numpy(out_face) )
    save_obj( args.save_obj, vertex_glasses_out,face_glasses.verts_idx )
    #shutil.copyfile('/Users/pierre/PhD/code_base/robust_gaze/data/45-oculos/glasses.png',os.path.join(os.path.dirname(args.save_obj),'glasses.png'))
    #os.cp('/Users/pierre/PhD/code_base/robust_gaze/data/45-oculos/glasses.png',os.path.join(os.path.dirname(args.save_obj),'glasses.png'))
    os.system('cp -r /Users/pierre/PhD/code_base/robust_gaze/data/45-oculos/ '+os.path.join(os.path.dirname(args.save_obj),'45-oculos'))
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    person_id = '00003900'
    define_path = lambda x,file: os.path.join('/Users/pierre/Downloads/EMOCA_v2_lr_mse_20/'+x+'/'+file)
    parser.add_argument('--obj_file_source', type=str, default='/Users/pierre/PhD/code_base/robust_gaze/data/reference_face_glasses/face_and_glasses_v2.obj')
    parser.add_argument('--obj_file_target', type=str, default=define_path(person_id,'mesh_coarse_trans.obj'))
    parser.add_argument('--obj_file_glasses', type=str, default='/Users/pierre/PhD/code_base/robust_gaze/data/45-oculos/oculos.obj')

    #parser.add_argument('--save_file', type=str, default='/Users/pierre/PhD/code_base/robust_gaze/output/glasses_fitted.obj')
    parser.add_argument('--save_obj', type=str, default=define_path(person_id,'oculus.obj'))
    args = parser.parse_args()
    main(args)


