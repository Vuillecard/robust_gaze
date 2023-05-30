import torch
import torchvision.transforms as T
from PIL import Image
from torch.nn.parallel import parallel_apply

import torchvision.transforms.functional as TF
import torch.nn.functional as F
import matplotlib.pyplot as plt

"""
Code to generate near and far focus blur.
code taken from : 
https://github.com/EPFL-VILAB/3DCommonCorruptions/blob/main/create_3dcc/create_dof.py
"""


def gaussian(M, std, sym=True, device=None):
    
    if M < 1:
        return torch.tensor([])
    if M == 1:
        return torch.ones((1,))
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(0, M, device=device) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]
        
    return w

def separable_gaussian(img, r=3.5, cutoff=None, device=None):
    
    if device is None:
        device = img.device
    if r < 1e-1:
        return img
    if cutoff is None:
        cutoff = int(r * 5)
        if (cutoff % 2) == 0: cutoff += 1

    assert (cutoff % 2) == 1
    img = img.to(device)
    _, n_channels, w, h = img.shape
    std = r
    fil = gaussian(cutoff, std, device=device).to(device)
    filsum = fil.sum() #/ n_channels
    fil = torch.stack([fil] * n_channels, dim=0)
    r_pad = int(cutoff) 
    r_pad_half = r_pad // 2
    #print(r_pad_half)
    img = F.pad(img, (r_pad_half, r_pad_half, r_pad_half, r_pad_half), "replicate", 0)  # effectively zero padding
    filtered = F.conv2d(img, fil.unsqueeze(1).unsqueeze(-2), bias=None, stride=1, padding=0, dilation=1, groups=n_channels)
    filtered /= filsum
    filtered = F.conv2d(filtered, fil.unsqueeze(1).unsqueeze(-1), bias=None, stride=1, padding=0, dilation=1, groups=n_channels)
    filtered /= filsum

    return filtered


def compute_circle_of_confusion(depths, aperture_size, focal_length, focus_distance):
    
    assert focus_distance > focal_length
    c = aperture_size * torch.abs(depths - focus_distance) / depths * (focal_length / (focus_distance - focal_length))
    
    return c

def compute_circle_of_confusion_no_magnification(depths, aperture_size, focus_distance):
        
    c = aperture_size * torch.abs(depths - focus_distance) / depths 
    
    return c

def compute_quantiles(depth, quantiles, eps=0.0001):
    
    depth_flat = depth.reshape(depth.shape[0], -1)
    quantile_vals = torch.quantile(depth_flat, quantiles, dim=1)
    quantile_vals[0] -= eps
    quantile_vals[-1] += eps
    
    return quantiles, quantile_vals

def compute_quantile_membership(depth, quantile_vals):
    
    quantile_dists = quantile_vals[1:]  - quantile_vals[:-1]
    depth_flat = depth.reshape(depth.shape[0], -1)
    calculated_quantiles = torch.searchsorted(quantile_vals, depth_flat)
    calculated_quantiles_left = calculated_quantiles - 1
    quantile_vals_unsqueezed = quantile_vals #.unsqueeze(-1).unsqueeze(-1)
    quantile_right = torch.gather(quantile_vals_unsqueezed, 1, calculated_quantiles).reshape(depth.shape)
    quantile_left = torch.gather(quantile_vals_unsqueezed, 1, calculated_quantiles_left).reshape(depth.shape)
    quantile_dists = quantile_right - quantile_left
    dist_right = ((quantile_right - depth) / quantile_dists) #/ quantile_dists[calculated_quantiles_left]
    dist_left = ((depth - quantile_left) / quantile_dists)  #/ quantile_dists[calculated_quantiles_left]
    
    return dist_left, dist_right, calculated_quantiles_left.reshape(depth.shape), calculated_quantiles.reshape(depth.shape)

def get_blur_stack_single_image(rgb, blur_radii, cutoff_multiplier):
    
    args = []
    
    for r in blur_radii:
        cutoff = None if cutoff_multiplier is None else int(r * cutoff_multiplier)
        if cutoff is not None and (cutoff % 2) == 0:
            cutoff += 1
        args.append((rgb, r, cutoff))
    
    blurred_ims = []
    
    blurred_ims = parallel_apply([separable_gaussian]*len(args), args)
    blurred_ims = torch.stack(blurred_ims, dim=1)
    
    return blurred_ims

def get_blur_stack(rgb, blur_radii, cutoff_multiplier=None):
    
    args = [(image.unsqueeze(0), radii, cutoff_multiplier) for image, radii in zip(rgb, blur_radii)]
    modules = [get_blur_stack_single_image for _ in args]
    outputs = []
    

    outputs = parallel_apply(modules, args)
    
    return torch.cat(outputs, dim=0)


def composite_blur_stack(blur_stack, dist_left, dist_right, values_left, values_right):
    
    shape = list(blur_stack.shape)
    shape[2] = 1
    composite_vals = torch.zeros(shape, dtype=torch.float32, device=blur_stack.device)
    sim_left = (1 - dist_left**2)
    sim_right = (1 - dist_right**2)
    
    _ = composite_vals.scatter_(1, index=values_left.unsqueeze(1).unsqueeze(2), src=sim_left.unsqueeze(1).unsqueeze(2))
    _ = composite_vals.scatter_(1, index=values_right.unsqueeze(1).unsqueeze(2), src=sim_right.unsqueeze(1).unsqueeze(2))
    
    composite_vals /= composite_vals.sum(dim=1, keepdims=True)
    composited = composite_vals * blur_stack
    composited = composited.sum(dim=1)
    
    return composited

def refocus_image(rgb, depth, focus_distance, aperture_size, quantile_vals, return_segments=False):
    quantile_vals_squeezed = quantile_vals.squeeze()
    dist_left, dist_right, calculated_quantiles_left, calculated_quantiles = compute_quantile_membership(depth, quantile_vals)
    blur_radii = compute_circle_of_confusion_no_magnification(quantile_vals, aperture_size, focus_distance)  
    #print(blur_radii)
    blur_stack = get_blur_stack(rgb, blur_radii, cutoff_multiplier=3)
    composited = composite_blur_stack(blur_stack, dist_left, dist_right, calculated_quantiles_left, calculated_quantiles)

    if return_segments:
        return composited, calculated_quantiles_left
    else:
        return composited
    
def replicate_batch(batch):
        
    batch = batch.unsqueeze(1)

    batch = torch.cat((batch, batch), dim = 1)
    
    return batch

def sample_idxs_2(quantiles):
    near = torch.arange(len(quantiles))[:1] #HARDCODED
    near = near[torch.randperm(len(near))[0]]
    
    far = torch.arange(len(quantiles))[7:8] #HARDCODED
    far = far[torch.randperm(len(far))[0]]
    
    return torch.stack((near, far))



def load_rgb_depth(image_loc, depth_loc):
    rgb = TF.to_tensor(Image.open(image_loc).convert("RGB"))
    depth = TF.to_tensor(Image.open(depth_loc))
    depth_normalized = depth.float()/(2.0**16)
    min_depth = depth_normalized[depth_normalized > 0.1].min()
    max_depth = depth_normalized.max() 
    depth_normalized = (depth_normalized - min_depth) / (max_depth - min_depth)
    depth_normalized = 0.9*depth_normalized + 0.1
    depth_normalized[depth_normalized <= 0.1] = 0.1 # set min depth to 0.15 as a background
    depth_normalized[depth_normalized >= 1.0] = depth_normalized[depth_normalized < 1.0].max()
    rgb = rgb[None,...] # [1,3,h,w]
    return rgb, depth

def wrapper_focus_blur(rgb, depth, focus_dists_idx = 3):

    """ Function to apply defocus blur to a single image using normalized depth map,
        focus_dists_idx from 0 focus far to 8 focus near
    """
    if isinstance(rgb, str) and isinstance(depth, str):
        rgb, depth = load_rgb_depth(rgb, depth)
    else : 
        n, c, h, w = rgb.shape
        assert n == 1, 'batch size must be 1'
        assert c == 3, 'rgb must have 3 channels'
        assert depth.shape == (1, 1, h, w), 'depth must have 1 channel'
        assert depth.max() <= 1.0, 'depth must be normalized'
        assert depth.min() >= 0.0, 'depth must be normalized'

    n_quantiles = 8 #hard-coded!!
    assert focus_dists_idx <= n_quantiles, 'focus_dists_idx must be less than n_quantiles'

    device = 'cpu'

    quantiles = torch.arange(0, n_quantiles + 1, device = device) / n_quantiles
    depth_normalized = depth            
    
    quantiles, quantile_vals = compute_quantiles(depth, quantiles, eps = 0.0001)
    quantile_vals = quantile_vals.permute(1, 0)
    
    print(quantile_vals.shape)
    if focus_dists_idx >= 2:
        aperture =  10
    else:
        aperture =  3  #hard-coded!!
    
    focus_dists = quantile_vals[:,focus_dists_idx]

    copies_to_return = 1
    apertures = torch.tensor([[aperture]] * copies_to_return, dtype = torch.float32, device = device)
    #apertures[1] = (apertures[1] - 2.5) / 2. #reduce aperture for far focus
    #apertures[0] = (apertures[0] - 1.5) / 1. #reduce aperture for far focus
    
    image_out= refocus_image(rgb, depth_normalized, focus_dists, apertures, quantile_vals)

    return image_out[0]
   