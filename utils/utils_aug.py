import imp

import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import skimage.morphology
import torch
import torchvision

from utils.utils_img import show_seg


def affine_grid(img, rotation=True, translation=True, scale=True):
    # rotate and tranlate batch
    # rotation_sigma = 2*np.pi*#/360*6/4
    rotation = 2 * np.pi * 0.02 if rotation else 0
    translation = 0.1 if translation else 0
    scale = 0.2 if scale else 0
    affines = []
    r_s = np.random.uniform(-rotation, rotation, img.shape[0])
    t_s = np.random.uniform(-translation, translation, img.shape[0])
    s_s = np.random.uniform(-scale, scale, img.shape[0]) + 1
    for r, t, s in zip(r_s, t_s, s_s):
        # convert origin to center
        # T1 = np.array([[1,0,1],[0,1,1],[0,0,1]])
        # scale
        S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])
        # rotation
        R = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
        # translation
        T = np.array([[1, 0, t], [0, 1, t], [0, 0, 1]])
        # convert origin back to corner
        # T2 = np.array([[1,0,-1],[0,1,-1],[0,0,1]])
        # M = T2@T@R@T1
        M = T @ R @ S  # the center is already (0,0), no need to T1, T2
        affines.append(M[:-1])
    M = np.stack(affines, 0)
    M = torch.tensor(M, dtype=img.dtype, device=img.device)
    grid = torch.nn.functional.affine_grid(M, size=img.shape, align_corners=False)
    return grid


def bspline_grid(img):
    # rotate and tranlate batch
    scale = 30
    grid = (torch.rand(img.shape[0], 2, 9, 9, device=img.device, dtype=img.dtype) - 0.5) * 2 / scale
    grid = torch.nn.functional.interpolate(grid, size=img.shape[2:], align_corners=False, mode="bicubic")
    grid = grid.permute(0, 2, 3, 1).contiguous()
    return grid


def augment(image, label=None, rigid=True, bspline=True):
    assert rigid == True
    with torch.no_grad():
        grid = affine_grid(image)
        if bspline:
            grid = grid + bspline_grid(image)
        image = torch.nn.functional.grid_sample(image, grid, padding_mode="reflection", align_corners=False, mode="bilinear")
        if label is not None:
            # print(label.shape)
            label = torch.nn.functional.grid_sample(label[:, None,...].type(torch.float32), grid, padding_mode="zeros", align_corners=False, mode="nearest")
            label = label.type(torch.long)[:, 0,...]
            # print(label.shape)
            # show_seg(image[:, 0,...], label, 'test_aug.png')
            # exit()
            return image, label
        else:
            return image
