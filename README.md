# robust_gaze

Robust gaze data augmentation for deep learning based gaze estimation.


## Installation

To run this code we need first to create a conda environment with the required dependencies. To do so, run the following commands:

```bash
conda create -n face_3d_aug python=3.9
conda activate face_3d_aug
```

for easy import you can add the path to the project path to the conda python path:
```bash
conda develop /path/to/robust_gaze_directory
```


Then, install the dependencies: 
- follow instruction to install [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
- On the same pytroch3d installation page install pytorch3d as proposed.
- then install nastort, skimage, PIL, matplotlib and numpy.

Regarding, the face 3d model please refer to the installation of emoca model [here](https://github.com/radekd91/emoca#installation)


## How to run the code

in the /demo folder you can find demo script of how to apply augmentation on a single image, batch of image or video for dynamic augmentation. 
in robust_gaze/object_list you can find the list of objects and texture  used for augmentation.