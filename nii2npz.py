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

try:
    os.makedirs("./data/cache/slices")
    os.remove("./data/cache/slices/*")
except:
    pass

cases = os.listdir(seg_path)
for case in tqdm(cases):

    org_data = sitk.ReadImage(f"{org_path}/{case}")
    org_data = sitk.DICOMOrient(org_data, "PIL")
    org_img = sitk.GetArrayFromImage(org_data).astype(np.float32)
    org_img = resize(org_img, 256, True)
    org_img = normalize(org_img, 1., 0.)
    org_img = org_img.type(torch.float16)
    assert org_img.shape[1] == org_img.shape[2] == 256

    seg_data = sitk.ReadImage(f"{seg_path}/{case}")
    seg_data = sitk.DICOMOrient(seg_data, "PIL")
    seg_img = sitk.GetArrayFromImage(seg_data).astype(np.int8)
    seg_img = resize(seg_img, 256, False)
    assert seg_img.shape[1] == seg_img.shape[2] == 256

    # show_seg(org_img, seg_img)

    n = org_img.shape[0]
    for i in range(n):
        np.savez_compressed(f"{out_patch}/{case}_{i}.npz", image=org_img[i], label=seg_img[i])
    # exit()
