import os 
from robust_gaze.utils.object_utils import load_obj_file, wrapper_find_transform,apply_transform
from typing import List,Union
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
        self.objects_mesh = []
        for obj in self.objects_name:
            object_mesh = load_obj_file(os.path.join(self.dir_3d_object,obj,obj+'_template.obj'))
            self.objects_mesh.append(object_mesh)

    def blur(self,):
        pass

    def render(self,):
        pass

    def process(self, image ,face_obj ):

        # find the transformation between template 3d face and predicted 3d face
        transformation = wrapper_find_transform(face_obj)
        # apply the transformation to the 3d object
        objects_fitted = [ apply_transform(obj_mesh,transformation) for obj_mesh in self.objects_mesh]
        
        # render the 3d object on the image


        pass