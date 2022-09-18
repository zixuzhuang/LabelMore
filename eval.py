import argparse
import os

import numpy as np
import SimpleITK as sitk
import torch

from utils.utils_img import normalize, resize, show_seg, show_slices

size = [512, 512]
try:
    os.makedirs("./data/previews")
    os.remove("./data/previews/*")
except:
    pass
try:
    os.makedirs("./data/predictions")
    os.remove("./data/predictions/*")
except:
    pass


def save_label(mri_data, preds, save_path):
    seg_data = sitk.GetImageFromArray(preds.cpu().numpy().astype(np.uint8))
    seg_data.SetOrigin(mri_data.GetOrigin())
    seg_data.SetDirection(mri_data.GetDirection())
    seg_data.SetSpacing(mri_data.GetSpacing())
    sitk.WriteImage(seg_data, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LabelMore")
    parser.add_argument("-m", type=str)
    parser.add_argument("-n", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda")
    net = torch.load(args.m, map_location=device)
    net.eval()
    torch.set_grad_enabled(False)

    cases = list(set(os.listdir("./data/images")) - set(os.listdir("./data/segmentations")))
    cases.sort()

    n = len(cases) if len(cases) < 10 or args.n == 0 else args.n
    for case in cases[:n]:
        org_data = sitk.ReadImage(f"./data/images/{case}")
        org_data = sitk.DICOMOrient(org_data, "PIL")
        org_img = sitk.GetArrayFromImage(org_data).astype(np.float32)
        org_size = org_img.shape[1:]
        org_img = resize(org_img, size, True)
        org_img = normalize(org_img, 1.0, 0.0)
        image = torch.tensor(org_img, dtype=torch.float32).to(device)
        pred = net(image[:, None])
        pred = torch.argmax(pred, dim=1)
        # show_seg(org_img, pred.cpu().numpy(), num_cls=8, save_path=f"./data/previews/{case}.png")
        pred = resize(pred, org_size, False)
        save_label(org_data, pred, f"./data/predictions/{case}")
        print(case)

    # show_seg(image, pred, "pred.png", num_cls=3)
    # exit()
