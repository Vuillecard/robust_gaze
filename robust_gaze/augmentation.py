import os 
from utils.object_utils import load_obj_file, wrapper_find_transform,apply_transform
from utils.render_image import get_render
from utils.focus_blur import wrapper_focus_blur
from utils.face_depth import wrapper_get_depth
from typing import List,Union
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import join_meshes_as_scene
"""
class for the augmentation of face in 3D
"""
DIR_NAME = os.path.dirname(__file__)

class Face3DAugmentation():

    def __init__(self,objects_name: List[str]):
        
        self.dir_3d_object = os.path.join(DIR_NAME,'object_list')
        self.objects_name = objects_name
        self.objects_mesh = None

        # pre load the selected object
        self.load_object()

    def load_object(self):
        self.objects_mesh = {}
        for obj in self.objects_name:
            object_mesh = load_objs_as_meshes([os.path.join(self.dir_3d_object,obj,obj+'_template.obj')])
            self.objects_mesh[obj] = object_mesh

    def blur(self,rgb,depth,focus_scale):
        print('rbg shape: ',rgb.shape)
        print('depth shape: ',depth.shape)
        return wrapper_focus_blur(rgb,depth,focus_scale)
        

    def process(self, image, face_obj, obj='glasses', lighting=((-1,0,0),), focus_id=None, return_depth=False):
        """
        focus_id: value between 0 to 8, 0 is far focus and 8 is near focus
        """
        # if lighting is None:
        #     lighting = (0,0,-1) # default lighting

        print('processing object: ',obj)
        # find the transformation between template 3d face and predicted 3d face
        transformation = wrapper_find_transform(face_obj)
        # apply the transformation to the 3d object
        if obj is not None:
            obj_mesh = self.objects_mesh[obj]
            object_fitted = apply_transform(obj_mesh, transformation) 
        else:
            object_fitted = None
        
        # render the 3d object along with lighting on the image
        if obj=='glasses':
            specular_color = [[1,1,1]]
        else:
            specular_color = None
        rendered_image = get_render(image, face_obj, object_fitted, lighting, specular_color)
        
        # apply depth of field effect
        if focus_id is not None:
            if obj is not None:
                face_obj = join_meshes_as_scene([face_obj, object_fitted])

            depth = wrapper_get_depth(face_obj)
            rendered_image = self.blur(rendered_image,depth, focus_id)
        
        if return_depth:
            return rendered_image, depth

        return rendered_image