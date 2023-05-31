import torch
import os
import argparse
import glob
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from pytorch3d.io import load_objs_as_meshes

from augmentation import Face3DAugmentation

def main(args):
        
    # create augmentor object
    object_names = ['glasses', 'mask', 'hat']
    Augmentor = Face3DAugmentation(object_names)
    
    # iterate over samples
    vis = False

    sample_name = os.path.basename(args.input_dir)
    print('processing sample: ', sample_name)
    image = cv2.imread(os.path.join(args.input_dir, 'inputs.png'))
    face_mesh = load_objs_as_meshes([os.path.join(args.input_dir, 'mesh_coarse.obj')])
    image_aug = Augmentor.process(image, face_mesh, 'glasses')
    print( 'image_aug.shape: ', image_aug.shape)
    # save augmented image
    if vis:
        plt.imshow(image_aug)
        plt.show()
    os.makedirs(args.output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(args.output_dir, sample_name+'_aug.png'), image_aug)
        
        
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help="directory of emoca outputs")
    parser.add_argument('--output_dir', type=str, help="output directory for augmented images")
    args = parser.parse_args()
    
    main(args)