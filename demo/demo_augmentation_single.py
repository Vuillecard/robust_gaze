import torch
import os
import argparse
import glob
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from pytorch3d.io import load_objs_as_meshes

from robust_gaze.augmentation import Face3DAugmentation

def main(args):
        
    # create augmentor object, pre load 3d object for faster processing
    object_names = ['glasses', 'mask', 'hat']
    Augmentor = Face3DAugmentation(object_names)
    
    # iterate over samples
    vis = False

    sample_name = os.path.basename(args.input_dir)
    print('processing sample: ', sample_name)
    image = cv2.imread(os.path.join(args.input_dir, 'inputs.png'))
    face_mesh = load_objs_as_meshes([os.path.join(args.input_dir, 'mesh_coarse_trans.obj')])
    image_aug = Augmentor.process(  image, # input image same used by emoca 
                                    face_mesh, # 3d face model from emoca
                                    obj = None, # select object to augment glasses, mask, hat
                                    lighting=((-0.5,0.,-0.5),), # set lighting direction (0,0,-1) is frontal lighting
                                    focus_id=7, # set focus level from 0 to 8, correspond to far focus to near focus
                                    return_depth=False # return depth map
                                    )
    
    # save augmented image
    if vis:
        plt.imshow(image_aug)
        plt.show()
    os.makedirs(args.output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(args.output_dir, sample_name+'_aug_dof_3_no_glasses.png'), image_aug.astype('uint8'))
    #cv2.imwrite(os.path.join(args.output_dir, sample_name+'_depth_3.png'), (depth*254).astype('uint8'))

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default = '',
                        help="directory of emoca outputs")
    parser.add_argument('--output_dir', type=str,default = '',
                         help="output directory for augmented images")
    args = parser.parse_args()
    
    main(args)