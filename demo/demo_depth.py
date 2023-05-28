import cv2 
import os 


def print_stat(img):
    print(img.min(), img.max(),img.mean(),img.std())

def main():

    dir_path = '/Users/pierre/Downloads/depth'

    for file in os.listdir(dir_path):
        if file.endswith('.png'):
            img = cv2.imread(os.path.join(dir_path,file))
            print(img.shape)
            print(img[45,39,:])
            print((img[:,:,0].tolist()))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            print(img.shape)
            print_stat(img)
            cv2.imwrite(file,img*255)

if __name__ == '__main__':
    main()