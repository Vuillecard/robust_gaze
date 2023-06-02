import cv2 
import os 
import natsort


def generate_video(image_folder):
    """
    Generates a video from a folder of images.
    image_folder: folder containing images
    """

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = natsort.natsorted(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(os.path.join(image_folder,'video.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 24, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":

    # example of usage
    image_folder = 'path/to/folder/with/images'
    generate_video(image_folder)
