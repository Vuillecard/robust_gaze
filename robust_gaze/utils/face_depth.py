import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.io import imread
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes
import robust_gaze.utils.face_depth_utils as util


def wrapper_get_depth(face_mesh, img_size=224):

    renderer = SRenderY(img_size, face_mesh, uv_size=256)
    trans_verts = face_mesh.verts_list()[0]
    op_depth = renderer.render_depth(trans_verts.unsqueeze(0))
    depth_img = op_depth[0].permute(1,2,0)
    depth_img = depth_img.squeeze().float().numpy()

    return depth_img

"""
This code is taken from :
https://github.com/radekd91/emoca/blob/release/EMOCA_v2/gdl/models/Renderer.py
"""

class Pytorch3dRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogenous
    Meshes.
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin': None,
            'perspective_correct': False,
        }
        raster_settings = util.dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        raster_settings = self.raster_settings
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        # pix_to_face(N,H,W,K), bary_coords(N,H,W,K,3),attribute: (N, nf, 3, D)
        # pixel_vals = interpolate_face_attributes(fragment, attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1]))
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        return pixel_vals


class SRenderY(nn.Module):
    def __init__(self, image_size, mesh_object, uv_size=256):
        super(SRenderY, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size
        
        verts = mesh_object.verts_list()[0]
        faces = mesh_object.faces_list()[0]
        #textures = mesh_object.textures_list()[0]
#         uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        #uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        #uvfaces = textures
        #faces = faces.verts_idx[None, ...]

        self.rasterizer = Pytorch3dRasterizer(image_size)
        self.uv_rasterizer = Pytorch3dRasterizer(uv_size)

        # faces
        #dense_triangles = util.generate_triangles(uv_size, uv_size)
        #self.register_buffer('dense_faces', torch.from_numpy(dense_triangles).long()[None, :, :])
        self.register_buffer('faces', faces)

        # shape colors, for rendering shape overlay
        colors = torch.tensor([255, 255, 255])[None, None, :].repeat(1, faces.max() + 1, 1).float() / 255.
        #face_colors = util.face_vertices(colors, faces)
        #self.register_buffer('face_colors', face_colors)

        ## SH factors for lighting
        pi = np.pi
        constant_factor = torch.tensor(
            [1 / np.sqrt(4 * pi), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), \
             ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), \
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi)))]).float()
        self.register_buffer('constant_factor', constant_factor)


    def render_depth(self, transformed_vertices):
        '''
        -- rendering depth
        '''
        batch_size = transformed_vertices.shape[0]

        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] - transformed_vertices[:, :, 2].min()
        z = -transformed_vertices[:, :, 2:].repeat(1, 1, 3)
        z = z - z.min()
        z = z / z.max()
        
        # Attributes
        attributes = util.face_vertices(z, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        depth_images = rendering[:, :1, :, :]
        return depth_images


if __name__=='__main__':
    
    ###### get depth map
    face_obj_file = None
    renderer = SRenderY(224, face_obj_file, uv_size=256)
    trans_verts,_,_ = load_obj(face_obj_file)
    op_depth = renderer.render_depth(trans_verts.unsqueeze(0))
    depth_img = op_depth[0].permute(1,2,0)
    depth_img = depth_img.squeeze().numpy()
    depth_img = depth_img*65536
    depth_img = depth_img.astype(np.uint16)