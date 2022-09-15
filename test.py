import numpy as np
import SimpleITK as sitk
import torch

from net.dataloader import Transform
from utils.utils_img import normalize, resize, show_seg, show_slices

size = 512
org_data = sitk.ReadImage("data/images/00011.nii.gz")
org_img = torch.tensor(sitk.GetArrayFromImage(org_data), dtype=torch.float32)
org_img = resize(org_img, size, True)
org_img = normalize(org_img, 1.0, 0.0)
org_img = org_img.type(torch.float16)[:, None].to("cuda")

seg_data = sitk.ReadImage("data/segmentations/00011.nii.gz")
seg_img = torch.tensor(sitk.GetArrayFromImage(seg_data).astype(np.uint8), dtype=torch.long)
seg_img = resize(seg_img, size, False)
seg_img = seg_img.type(torch.float16)[:, None].to("cuda")
print(torch.max(seg_img))

t = Transform("cuda")
org, seg = t(org_img, seg_img)
print(org.dtype, seg.dtype)
print(torch.max(seg))
show_seg(org, seg, num_cls=9)
