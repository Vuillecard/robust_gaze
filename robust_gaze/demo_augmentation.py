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
    vis = True
    files = glob.glob(os.path.join(args.input_dir, '00*'))
   
    for sample in tqdm(files):
        
        sample_name = os.path.basename(sample)
        image = cv2.imread(os.path.join(sample, 'inputs.png'))
        face_mesh = load_objs_as_meshes([os.path.join(sample, 'mesh_coarse.obj')])
        image_aug = Augmentor.process(image, face_mesh, 'glasses')
        
        # save augmented image
        if vis:
            plt.imshow(image_aug)
            plt.show()
        cv2.imwrite(os.path.join(args.output_dir, sample_name+'.png'), image_aug)
        
        
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help="directory of emoca outputs")
    parser.add_argument('--output_dir', type=str, help="output directory for augmented images")
    args = parser.parse_args()
    
    main(args)