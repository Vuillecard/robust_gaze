
import os
import torch
import matplotlib.pyplot as plt
import numpy as np 
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj,load_ply
import cv2
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    look_at_view_transform,
    FoVOrthographicCameras,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights,
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.renderer.lighting import diffuse
# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Set paths
DATA_DIR = "./data"
#obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")
#obj_filename = '/Users/pierre/PhD/code_base/robust_gaze/data/reference_face_glasses/face_and_glasses_v2.obj'
#obj_filename = '/Users/pierre/PhD/code_base/robust_gaze/data/test_fitted_glasses/mesh_coarse_transformed_v3.obj'
#obj_filename = '/Users/pierre/PhD/code_base/robust_gaze/data/reference_face_glasses/faces_emoca/face_1/mesh_coarse.obj'
#obj_filename = '/Users/pierre/PhD/code_base/robust_gaze/demo/data/mesh_coarse_transformed.obj'


person_id = '00023100' # 00003900

image_og_path = '/Users/pierre/Downloads/EMOCA_v2_lr_mse_20/'+person_id+'/inputs.png'
# #obj_filename ='/Users/pierre/Downloads/EMOCA_v2_lr_mse_20/00003900/mesh_coarse.obj'
obj_filename ='/Users/pierre/Downloads/EMOCA_v2_lr_mse_20/'+person_id+'/mesh_coarse_trans.obj'
# #obj_filename = '/Users/pierre/Downloads/EMOCA_v2_lr_mse_20/00003900/mesh_coarse_trans_detail.obj'

# Load obj file
#mesh = load_objs_as_meshes([obj_filename], device=device)


# add path for demo utils functions 

def load_mesh(filename):
    fname, ext = os.path.splitext(filename)
    if ext == '.ply':
        vertices, faces = load_ply(filename)
    elif ext == '.obj':
        vertices, face_data, _ = load_obj(filename)
        faces = face_data[0]
    else:
        raise ValueError("Unknown extension '%s'" % ext)
    return vertices, faces

verts, faces = load_mesh(obj_filename)

verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)

# verts_rgb[:, :, 0] = 135 / 255
# verts_rgb[:, :, 1] = 206 / 255
# verts_rgb[:, :, 2] = 250 / 255
#verts_rgb = torch.ones_like(verts)[None]/255  # (1, V, 3)
#
# verts_rgb[:,:,0] = 30/255
# verts_rgb[:,:,1] = 206/255
# verts_rgb[:,:,2] = 250/255

# verts_rgb[:,:,0] = 0/255
# verts_rgb[:,:,1] = 191/255
# verts_rgb[:,:,2] = 255/255

textures = TexturesVertex(verts_features=verts_rgb.to(device))
mesh = Meshes([verts ], [faces ], textures)

# Initialize a camera.
# With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
# So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 

#azim = torch.linspace(-90, 90, 1)
#R, T = look_at_view_transform(0.35, elev=0, azim=azim, at=((0, -0.025, 0),), )
#cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
#print(0.03*180/np.pi, 0.02*180/np.pi)
R, T = look_at_view_transform(10,0,0)
T *= -1
R = R@torch.Tensor([[[1,  0.,  0.],
         [ 0.,  -1.,  0.],
         [ 0.,  0., -1.]]])
print(R, T)
cameras = FoVOrthographicCameras(device=device, R=R, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=1025, 
    blur_radius=0.0, 
    faces_per_pixel=1)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
# -z direction. 

# we use only diffuse lighting as described in the paper changing light 
lights = DirectionalLights(device=device, direction=((1, 1, -1),),
                             ambient_color=((0.0, 0.0, 0.0),),
                             diffuse_color=((1., 1., 1.),),
                             specular_color=((1.0, 1.0, 1.0),)
                             )
# lights = PointLights(device=device, location=((10, 1, -10),),
#                              ambient_color=((0.0, 0.0, 0.0),),
#                              diffuse_color=((1., 1., 1.),),
#                              specular_color=((0.0, 0.0, 0.0),)
#                              )
#lights = pytroch3d.Lights(device=device, ambient_color=((0.5, 0.5, 0.5),),
#lights = PointLights(device=device, location=[[100, 0.0,-10]])

# Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
# interpolate the texture uv coordinates for each vertex, sample from a texture image and 
# apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)
materials = Materials(
            device=device,
            specular_color=[[1.0, 1.0, 1.0]],
            shininess=20
        )
images = renderer(mesh,materials=materials)
print(images.shape)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.savefig('render_test.png')

img_og = cv2.imread(image_og_path)

# load original image 
ligth = images[0,...,:3].cpu().numpy()
ligth = cv2.resize(ligth, (img_og.shape[0],img_og.shape[1]))
#ligth = cv2.cvtColor(ligth, cv2.COLOR_BGR2GRAY)
rescaled_light = (ligth - ligth.min()) / (ligth.max() - ligth.min())

#img_out_multiply = img_og*rescaled_light.repeat(3).reshape(img_og.shape)
img_out_multiply = img_og*ligth


hsv_image = cv2.cvtColor(img_og, cv2.COLOR_BGR2HLS)
hsv_image[...,1] = hsv_image[...,1]*cv2.cvtColor(ligth, cv2.COLOR_BGR2HLS)[...,1]
img_out_hsv = cv2.cvtColor(hsv_image, cv2.COLOR_HLS2BGR)

cv2.imwrite('og.png', img_og)
cv2.imwrite('render_light.png', rescaled_light*255)
cv2.imwrite('render_og_multiply.png', img_out_multiply)
cv2.imwrite('render_og_hsv.png', img_out_hsv)


