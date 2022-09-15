import os

# import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

from utils.utils_img import normalize, resize, show_nii, show_seg

org_path = "./data/images"
seg_path = './data/segmentations'
out_patch = "./data/cache/slices"
size = 512
num_cls = 9

try:
    os.makedirs("./data/cache/slices")
    os.remove("./data/cache/slices/*")
except:
    pass

cases = os.listdir(seg_path)
for case in tqdm(cases):

    org_data = sitk.ReadImage(f"{org_path}/{case}")
    org_img = sitk.GetArrayFromImage(org_data).astype(np.float32)
    org_img = resize(org_img, size, True)
    org_img = normalize(org_img, 1., 0.)
    org_img = org_img.type(torch.float16)
    assert org_img.shape[1] == org_img.shape[2] == size

    seg_data = sitk.ReadImage(f"{seg_path}/{case}")
    seg_img = sitk.GetArrayFromImage(seg_data).astype(np.int8)
    seg_img = resize(seg_img, size, False)
    assert seg_img.shape[1] == seg_img.shape[2] == size

    # show_seg(org_img, seg_img, num_cls=num_cls)
    # exit()

    n = org_img.shape[0]
    for i in range(n):
        np.savez_compressed(f"{out_patch}/{case}_{i}.npz", image=org_img[i], label=seg_img[i])
    # exit()
