import numpy as np
import torch
import os 
from typing import Dict

try:
    from pytorch3d.io import load_objs_as_meshes, load_obj
    from pytorch3d.structures import Meshes
except:
    print("pytorch3d not installed, some functions will not work")

"""
This file contains the functions to compute the transformation between two 3d objects, 
and handle some usful operations on 3d objects:
    fit_3d_object
    compute_inverse_transform
"""
FIT_SUPPORT_INDEX = [3526,2736,1600,3542,1649,2012,1962,3908,3173,2252,785]
DIR_3D_OBJECT = os.path.join(os.path.dirname(os.path.dirname(__file__)),'object_list')


def wrapper_find_transform(target):
    """ 
    wrapper function to compute the transformation between two 3d objects for EMOCA 3d face
    """
    if not torch.is_tensor(target): 
        target = torch.tensor(target)

    path_head_template = os.path.join(DIR_3D_OBJECT,'head_template','head_template.obj')
    source ,_= load_obj_file(path_head_template)

    transform_mat, scale = fit_3d_object(target.numpy()[FIT_SUPPORT_INDEX].T,source.numpy()[FIT_SUPPORT_INDEX].T)

    return { 'transform_mat': transform_mat, 'scale':scale}


def apply_transform(mesh_object : Meshes, transformation: Dict) -> Meshes:
    """
    apply the transformation to a pytorth3d mesh object, and return the transformed mesh object
    """
    transform_mat = transformation['transform_mat']
    object_verts = mesh_object.verts_list()[0]
    object_faces = mesh_object.faces_list()[0]
    # Apply the transformation to the glasses
    vertex_glasses_out = transformation['scale']*(object_verts@transform_mat[:3,:3].T + transform_mat[:3,3])

    # save the fitted object
    mesh_object_fitted = Meshes(verts=[vertex_glasses_out], faces=[object_faces], textures=mesh_object.textures)
    return mesh_object_fitted


def fit_3d_object(target,source):
    """
    Simple method to find the transformation from between two 3d object
    """
    vertex_weights = [1/len(source.T)]*len(source.T)
    vertex_weights = np.reshape(vertex_weights,(len(source.T),))
    lmks = target.copy()
   
    # rescale the landmarks to best match the canonical_metric_landmarks size
    first_iteration_scale = estimate_scale(source, lmks, vertex_weights)
    lmks[2, :] /= first_iteration_scale
    lmks[0, :] /= first_iteration_scale
    lmks[1, :] /= first_iteration_scale

    # Again rescale the landmarks to best match the canonical_metric_landmarks size
    second_iteration_scale = estimate_scale(source, lmks, vertex_weights)
    lmks[2, :] /= second_iteration_scale
    lmks[0, :] /= second_iteration_scale
    lmks[1, :] /= second_iteration_scale
    
    scale = first_iteration_scale*second_iteration_scale
    pose_transform_mat = solve_weighted_orthogonal_problem(
            source,
            lmks,
            vertex_weights
            )
    
    return pose_transform_mat, scale


def compute_inverse_transform(transform_mat):
    """ Simple function to compute the inverse of a transformation matrix"""
    inv_pose_transform_mat = np.linalg.inv(transform_mat)
    inv_pose_rotation = inv_pose_transform_mat[:3, :3]
    inv_pose_translation = inv_pose_transform_mat[:3, 3]
    return inv_pose_transform_mat, inv_pose_rotation, inv_pose_translation


def estimate_scale(source, target, vertex_weights):
    transform_mat = solve_weighted_orthogonal_problem(
            source,
            target,
            vertex_weights
            )
    return np.linalg.norm(transform_mat[:, 0])


def extract_square_root(point_weights):
    return np.sqrt(point_weights)


def solve_weighted_orthogonal_problem(source_points, target_points,
                                      point_weights):
    sqrt_weights = extract_square_root(point_weights)
    transform_mat = internal_solve_weighted_orthogonal_problem(
            source_points,
            target_points,
            sqrt_weights
            )
    return transform_mat

def load_obj_file(obj_filename):
    """
    Load the obj file as a pytorch3d object
    """
    verts, faces_idx, _ = load_obj(obj_filename)
    return verts, faces_idx

def internal_solve_weighted_orthogonal_problem(sources, targets, sqrt_weights):

    # tranposed(A_w).
    weighted_sources = sources * sqrt_weights[None, :]

    # tranposed(B_w).
    weighted_targets = targets * sqrt_weights[None, :]

    # w = tranposed(j_w) j_w.
    total_weight = np.sum(sqrt_weights * sqrt_weights)

    # Let C = (j_w tranposed(j_w)) / (tranposed(j_w) j_w).
    # Note that C = tranposed(C), hence (I - C) = tranposed(I - C).
    #
    # tranposed(A_w) C = tranposed(A_w) j_w tranposed(j_w) / w =
    # (tranposed(A_w) j_w) tranposed(j_w) / w = c_w tranposed(j_w),
    #
    # where c_w = tranposed(A_w) j_w / w is a k x 1 vector calculated here:
    twice_weighted_sources = weighted_sources * sqrt_weights[None, :]
    source_center_of_mass = np.sum(twice_weighted_sources, axis=1) / total_weight

    # tranposed((I - C) A_w) = tranposed(A_w) (I - C) =
    # tranposed(A_w) - tranposed(A_w) C = tranposed(A_w) - c_w tranposed(j_w).
    centered_weighted_sources = weighted_sources - np.matmul(
            source_center_of_mass[:, None], sqrt_weights[None, :])

    design_matrix = np.matmul(weighted_targets, centered_weighted_sources.T)

    rotation = compute_optimal_rotation(design_matrix)

    scale = compute_optimal_scale(
            centered_weighted_sources,
            weighted_sources,
            weighted_targets,
            rotation
            )

    rotation_and_scale = scale * rotation

    pointwise_diffs = weighted_targets - np.matmul(rotation_and_scale,
                                                   weighted_sources)

    weighted_pointwise_diffs = pointwise_diffs * sqrt_weights[None, :]

    translation = np.sum(weighted_pointwise_diffs, axis=1) / total_weight

    transform_mat = combine_transform_matrix(rotation_and_scale, translation)

    return transform_mat


def compute_optimal_rotation(design_matrix):
    if np.linalg.norm(design_matrix) < 1e-9:
        print("Design matrix norm is too small!")

    u, _, vh = np.linalg.svd(design_matrix, full_matrices=True)

    postrotation = u
    prerotation = vh

    if np.linalg.det(postrotation) * np.linalg.det(prerotation) < 0:
        postrotation[:, 2] = -1 * postrotation[:, 2]

    rotation = np.matmul(postrotation, prerotation)

    return rotation


def compute_optimal_scale(centered_weighted_sources, weighted_sources,
                          weighted_targets, rotation):
    rotated_centered_weighted_sources = np.matmul(
            rotation, centered_weighted_sources)

    numerator = np.sum(rotated_centered_weighted_sources * weighted_targets)
    denominator = np.sum(centered_weighted_sources * weighted_sources)

    if denominator < 1e-9:
        print("Scale expression denominator is too small!")
    if numerator / denominator < 1e-9:
        print("Scale is too small!")

    return numerator / denominator


def combine_transform_matrix(r_and_s, t):
    result = np.eye(4)
    result[:3, :3] = r_and_s
    result[:3, 3] = t
    return result