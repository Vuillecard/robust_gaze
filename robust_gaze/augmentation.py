import os 
from utils.object_utils import load_obj_file, wrapper_find_transform,apply_transform
from utils.render_image import get_render
from typing import List,Union
from pytorch3d.io import load_objs_as_meshes
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

    def blur(self,):
        pass

    def process(self, image, face_obj, obj='glasses', lighting=((-1,0,0),), specular_color=None):

        # find the transformation between template 3d face and predicted 3d face
        transformation = wrapper_find_transform(face_obj)
        # apply the transformation to the 3d object
        obj_mesh = self.objects_mesh[obj]
        object_fitted = apply_transform(obj_mesh, transformation) 
        
        # render the 3d object on the image
        if obj=='glasses':
            specular_color = [[1,1,1]]
        rendered_image = get_render(image, face_obj, object_fitted, lighting, specular_color)
        
        return rendered_image