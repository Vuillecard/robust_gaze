# robust_gaze

3D face data augmentation for deep learning based gaze estimation robustness improvement.


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
- Since pytorch3d have different configuration for different system please follow official instruction to install [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
- On the same pytroch3d installation page install pytorch3d as proposed.
- then install opencv, nastort, skimage, PIL, matplotlib and numpy.
```bash
pip install opencv-python
pip install nastort
pip install scikit-image
pip install Pillow
pip install matplotlib
pip install numpy
```

Regarding, the face 3d model EMOCA please refer to the installation of emoca model [here](https://github.com/radekd91/emoca#installation)

Moreover, the model used for the gaze prediction is the gaze360 model. Please refer to the installation of gaze360 model [here](https://github.com/erkil1452/gaze360)
## How to run the code

In the /demo folder you can find demo script of how to apply augmentation on a single image, batch of image or video for dynamic augmentation. 
in robust_gaze/object_list you can find the list of objects and texture  used for augmentation.

We put a sample example in /data/sample_example to run the code on a single image in the /demo folder. To run the code on a single image run the following command:

```bash
cd demo
python demo_single_image.py
```
The augmentation will be saved in the /output folder.

## Code structure

    .
    ├── Data                                # Sample for demonstration 
    ├── Demo                                # Demo code to run and understand the code 
    ├── Output                              # output if any
    ├── Robust_gaze                         # Source files 
        ├── object_list                     # List of object and texture used for augmentation
        ├── augmentation.py                 # Class for augmentation 
        └── Utils                           # Utility functions
            ├── face_depth.py               # script to generate face depth map
            ├── face_depth_utils.py         # utility functions for face depth map
            ├── focus_blur.py               # script to generate focus blur
            ├── object_utils.py             # utility functions for object augmentation
            ├── render_image.py             # script to render image
            ├── image.py                    # script to help handle image
            ├── video_utils.py              # script to help handle video
            └── visualize_utils.py          # script to help handle visualization    
    └── README.md

