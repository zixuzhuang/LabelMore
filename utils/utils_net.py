import copy
import csv
import logging
import os
import random
import time

import h5py
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch as th
import torchvision
from sklearn.model_selection import KFold
from torch.nn.functional import one_hot

from utils.utils_img import show_seg


def get_args(parser):
    parser.add_argument("-c", type=str, default="config/unet.yaml")
    parser.add_argument("-t", type=bool, default=False)
    args = parser.parse_args()
    return args


def lr_func(epoch):
    warmup = 5
    period = 10
    decay = 0.97
    if epoch < warmup:
        lr = 1 / (warmup - epoch + 1)
        return lr
    else:
        lr_cos = (np.cos((epoch - warmup) / period * 2 * np.pi) + 1) / 2
        # lr_step = max((epoch - warmup + period // 2) // (period * interval) + 1, 1)
        lr_step = epoch - warmup
        lr = max(1 * lr_cos * decay ** lr_step, 1e-2)
        return lr


def initLogging(logFilename):
    """Init for logging"""
    logger = logging.getLogger("")

    if not logger.handlers:
        logging.basicConfig(
            level=logging.DEBUG,
            format="[%(asctime)s-%(levelname)s] %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
            filename=logFilename,
            filemode="w",
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s-%(levelname)s] %(message)s")
        console.setFormatter(formatter)
        logger.addHandler(console)


def init_train(cfg):
    log_dir = f"{time.strftime('%Y%m%d-%H%M%S')}.log"
    initLogging(log_dir)
    format = ("\n" + "|{:^9}" * 2 + "|") * 2
    title = ["BS", "LR"]
    items = [cfg.bs, cfg.lr]
    logging.info(format.format(*title, *items))
    return


def save_model(result, nets, cfg):
    # Save best model
    if result.best_epoch == result.epoch:
        for name, net in nets.items():
            torch.save(net, cfg.ckpt_path + ".pth")
    # for name, net in nets.items():
    #     torch.save(net, cfg.ckpt_path + f"_{info}last_{name}.pth")
    logging.info("BEST : {:.3f}, EPOCH: {:3}".format(result.best_result, result.best_epoch + 1))
    return


def dice_loss(preds, label, num_cls=7):
    """
    preds: [bs, c, H, W], torch.float32, c is the num of cls
    label: [bs, H, W], torch.long, label map
    dice:  [bs, c], torch.float32, dice of each cls
    """
    smooth = 1e-8
    bs = label.shape[0]
    preds = torch.argmax(preds, dim=1)
    true_one_hot = one_hot(label, num_cls).reshape(bs, -1, num_cls).type(torch.float32)
    pred_one_hot = one_hot(preds, num_cls).reshape(bs, -1, num_cls).type(torch.float32)
    intersection = torch.sum(true_one_hot * pred_one_hot)
    dice = (2.0 * intersection) / (torch.sum(pred_one_hot) + torch.sum(true_one_hot) + smooth)
    return 1 - torch.sum(dice)


def dividi_data(seed=2022, n=5):
    datatypes = ("train", "valid")
    data_path = "data/cache/slices"
    slices = np.array(os.listdir(data_path), dtype=str)
    slice_cases = [_.split("_")[0] for _ in slices]
    cases = list(set(slice_cases))
    kf = KFold(n_splits=n, shuffle=True, random_state=seed)
    index = {dt: [] for dt in datatypes}
    index["train"] = np.concatenate([list(kf.split(cases))[_][1] for _ in range(n - 1)])
    index["valid"] = list(kf.split(cases))[n - 1][1]
    for dt in datatypes:
        slice_idxs = []
        for case_idx in index[dt]:
            slice_idxs += [i for i, x in enumerate(slice_cases) if x == cases[case_idx]]
        index[dt] = slices[slice_idxs].tolist()
    return index
