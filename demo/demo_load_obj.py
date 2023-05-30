import os 
from robust_gaze.utils.object_utils import load_obj_file, fit_3d_object
from robust_gaze.utils.visualize_utils import plot_3d_vertices
from pytorch3d.structures import Meshes
from pytorch3d.io import IO ,load_objs_as_meshes

"""
    This script is used to load the 3d object and save it as a mesh in the right format
"""

def main_1():
    torch3d_manager = IO()
    device = 'cpu'
    dir_name = os.path.dirname(__file__)
    dir_3d_object = os.path.join(dir_name,'object_3d')

    dir_3d_new_object = os.path.join(dir_name,'object_list')
    OBJ_3D = { 'glasses' : os.path.join(dir_3d_object,'glasses_template','glasses_template.obj'),
            'cap' : os.path.join(dir_3d_object,'cap_template','cap_template.obj'),
            'mask': os.path.join(dir_3d_object,'mask_template','mask_template.obj')}


    for obj in ['glasses','cap','mask']:
        mesh_object = load_objs_as_meshes([OBJ_3D[obj]], device=device,load_textures=True)
        object_verts = mesh_object.verts_list()[0]
        object_faces = mesh_object.faces_list()[0]

        # save the fitted object
        new_mesh = Meshes(verts=[object_verts], faces=[object_faces], textures=mesh_object.textures)
        os.makedirs(os.path.join(dir_3d_new_object,obj),exist_ok=True)
        torch3d_manager.save_mesh(new_mesh,os.path.join(dir_3d_new_object,obj,obj+'_template.obj'))


def main_2():
    torch3d_manager = IO()
    device = 'cpu'
    dir_name = os.path.dirname(__file__)
    dir_3d_object = os.path.join(dir_name,'object_3d')

    dir_3d_new_object = os.path.join(dir_name,'object_list')
    OBJ_3D = { 'hat' : os.path.join(dir_3d_object,'hat','hat.obj')}

    for obj in ['hat']:
        mesh_object = load_objs_as_meshes([OBJ_3D[obj]], device=device,load_textures=True)
        object_verts = mesh_object.verts_list()[0]
        object_faces = mesh_object.faces_list()[0]

        # save the fitted object
        new_mesh = Meshes(verts=[object_verts], faces=[object_faces], textures=mesh_object.textures)
        os.makedirs(os.path.join(dir_3d_new_object,obj),exist_ok=True)
        torch3d_manager.save_mesh(new_mesh,os.path.join(dir_3d_new_object,obj,obj+'_template.obj'))


if __name__ == '__main__':
    
    main_2()