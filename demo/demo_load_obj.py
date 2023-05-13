import os 
from robust_gaze.utils.object_utils import load_obj_file, fit_3d_object
from robust_gaze.utils.visualize_utils import plot_3d_vertices


v,f = load_obj_file('/Users/pierre/PhD/code_base/robust_gaze/data/45-oculos/oculos.obj')
print(v.numpy().shape)
print(f.verts_idx)
print(f._fields)