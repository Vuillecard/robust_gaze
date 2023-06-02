import torch
import os
import argparse
import glob
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from pytorch3d.io import load_objs_as_meshes
from natsort import natsorted
from robust_gaze.augmentation import Face3DAugmentation
import numpy as np

"""
This script shows how to run dynamic augmentation on a video sequence
"""

def main(args):
        
    # create augmentor object
    object_names = ['glasses', 'mask', 'hat']
    Augmentor = Face3DAugmentation(object_names)
    
    # iterate over samples
    vis = False
    files = glob.glob(os.path.join(args.input_dir, '00*'))
    files = natsorted(files)
    os.makedirs(args.output_dir, exist_ok=True)

    # parameters for dynamic augmentation
    light_change = np.linspace(-1,1,120-15+1)
    focus_id_change = np.linspace(0,8,290-180+1,dtype=int)

    for sample in tqdm(files):
       
        sample_name = os.path.basename(sample)
        idx_frame = int(sample_name.split('_')[0])
        print('processing sample: ', sample_name, idx_frame)
        image = cv2.imread(os.path.join(sample, 'inputs.png'))

        if idx_frame <= 15 : 
            image_aug = image
        elif idx_frame <= 120 :
            face_mesh = load_objs_as_meshes([os.path.join(sample, 'mesh_coarse.obj')])
            image_aug = Augmentor.process(image, 
                                        face_mesh, 
                                        obj = None,
                                        lighting=((light_change[idx_frame-15],0,-np.sqrt(1-light_change[idx_frame-15]**2)),),
                                        focus_id=None)
        elif idx_frame <= 150 :
            face_mesh = load_objs_as_meshes([os.path.join(sample, 'mesh_coarse.obj')])
            image_aug = Augmentor.process(image, 
                                        face_mesh, 
                                        obj = 'mask',
                                        lighting=((1,0,0),),
                                        focus_id=None)
        elif idx_frame <= 180 :
            face_mesh = load_objs_as_meshes([os.path.join(sample, 'mesh_coarse.obj')])
            image_aug = Augmentor.process(image, 
                                        face_mesh, 
                                        obj = 'glasses',
                                        lighting=((1,0,0),),
                                        focus_id=None)
            
        elif idx_frame <= 290 : 
            face_mesh = load_objs_as_meshes([os.path.join(sample, 'mesh_coarse.obj')])
            image_aug = Augmentor.process(image, 
                                        face_mesh, 
                                        obj = 'glasses',
                                        lighting=((1,0,0),),
                                        focus_id=focus_id_change[idx_frame-180])
        # save augmented image
        if vis:
            plt.imshow(image_aug)
            plt.show()
        cv2.imwrite(os.path.join(args.output_dir, sample_name+'.png'), image_aug.astype('uint8'))
    

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help="directory of emoca outputs")
    parser.add_argument('--output_dir', type=str, help="output directory for augmented images")
    args = parser.parse_args()
    
    main(args)