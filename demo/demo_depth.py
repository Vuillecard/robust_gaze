import cv2 
import os 
import torchvision.transforms.functional as TF
from PIL import Image
import torch 
import matplotlib.pyplot as plt
def print_stat(img):
    print(img.min(), img.max(),img.mean(),img.std())

def main():

    dir_path = '/Users/pierre/Downloads/depth'

    for file in os.listdir(dir_path):
        if file.endswith('.png'):
            img = cv2.imread(os.path.join(dir_path,file),cv2.IMREAD_UNCHANGED)
            # normalize 
            img = img/(2**16)
            print(img.shape)
         
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            print_stat(img)
            cv2.imwrite(file,img*255)


def main_tensor():

    dir_path = '/Users/pierre/Downloads/depth'

    for file in os.listdir(dir_path):
        if file.endswith('.png'):
            img = TF.to_tensor(Image.open(os.path.join(dir_path,file)))
            img = img.float()/(2.0**16)
            img[img<0.15] = 0.15
            # normalize 
            print(img.shape)

            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            quantiles = torch.arange(0, 8 + 1) / 8
            print(quantiles)
            depth_flat = img.reshape((img.shape[0],-1))
            quantile_vals = torch.quantile(depth_flat, quantiles, dim=1)
            print_stat(img)
            cv2.imwrite(file,img[0].numpy()*255)
            
            plt.figure()
            plt.hist(depth_flat[0], density=True, bins=30)  # density=False would make counts
            plt.ylabel('Probability')
            plt.xlabel('Data')
            plt.savefig(file.replace('.png','_hist.png'))

if __name__ == '__main__':
    main_tensor()