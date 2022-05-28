import logging
import os
# import os
import time

import numpy as np
import torch
# import yaml
from torch.nn.functional import one_hot
from utils.utils_img import show_seg

from net.config import Config


def cal_dice(preds, label, num_cls=2):
    """
    preds: [bs, c, H, W], torch.float32, c is the num of cls
    label: [bs, H, W], torch.long, label map
    dice:  [bs, c], torch.float32, dice of each cls
    """
    smooth = 1e-8
    bs = label.shape[0]
    preds = torch.argmax(preds, dim=1)
    true_one_hot = one_hot(label, num_cls).reshape(-1, num_cls)
    pred_one_hot = one_hot(preds, num_cls).reshape(-1, num_cls)
    intersection = torch.sum(true_one_hot * pred_one_hot, dim=0)
    masksumarea = torch.sum(pred_one_hot, dim=0) + torch.sum(true_one_hot, dim=0)
    dice = (2.0 * intersection + smooth) / (masksumarea + smooth)
    return dice


class ResultSeg(object):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.epoch = 1
        self.best_epoch = 0
        self.best_result = 0.0
        return

    def add_dice(self, dice):
        self.dices.append(dice)
        return

    def init(self):
        self.st = time.time()
        self.imgs_show = None
        self.segs_show = None
        self.pred_show = None
        self.dices = []
        return

    def stastic(self):
        """
        in  self.dices: E [bs, c], torch.float32, c is the num of cls, E is the num of mini-batch
        out self.dices: [c]
        """
        self.dices = torch.stack(self.dices, dim=0)
        self.dices = torch.mean(self.dices, dim=0)
        self.time = np.round(time.time() - self.st, 1)
        return

    def print(self, datatype: str, epoch: int):
        dices = self.dices.reshape(-1)
        dice_title = []
        for i in range(self.cfg.num_cls):
            dice_title.append(f"D{i}")
        titles = ["dataset"] + dice_title
        items = [datatype.upper()] + [dices[_].item() for _ in range(self.cfg.num_cls)]
        forma_1 = "\n|{:^8}" + "|{:^5}" * (len(titles) - 1) + "|"
        forma_2 = "\n|{:^8}" + "|{:^.3f}" * (len(titles) - 1) + "|"
        logging.info(f"{datatype.upper()} DICE_MEAN: {np.round(torch.mean(dices).item(),2)}, TIME: {self.time}s")
        logging.info((forma_1 + forma_2).format(*titles, *items))
        self.epoch = epoch

        if torch.mean(dices) > self.best_result:
            self.best_epoch = epoch
            self.best_result = torch.mean(dices)
        return

    def add_img(self, imgs, segs, pred):
        """
        add img to 36 for show seg performance
        imgs: [2, N, H, W], list of 3 tensor
        segs: [3, N, H, W]
        pred: [N, H, W], list of 3 tensor
        """
        noise = torch.rand(imgs.shape[0], device=imgs.device)  # noise in [0, 1]
        ids = torch.argsort(noise, dim=0)[:2]
        pred = torch.argmax(pred, dim=1)
        if self.imgs_show is None:
            self.imgs_show = imgs[ids, 0]
            self.segs_show = segs[ids]
            self.pred_show = pred[ids]
        elif self.imgs_show.shape[0] < 36:
            self.imgs_show = torch.cat([self.imgs_show, imgs[ids, 0]], dim=0)
            self.segs_show = torch.cat([self.segs_show, segs[ids]], dim=0)
            self.pred_show = torch.cat([self.pred_show, pred[ids]], dim=0)
        elif self.imgs_show.shape[0] > 36:
            self.imgs_show = self.imgs_show[:36]
            self.segs_show = self.segs_show[:36]
            self.pred_show = self.pred_show[:36]
        return

    def show(self):
        show_seg(self.imgs_show, self.segs_show, os.path.join(self.cfg.log_dir.replace(".log", "_label.png")), nrow=6, num_cls=self.cfg.num_cls, alpha=0.6)
        show_seg(self.imgs_show, self.pred_show, os.path.join(self.cfg.log_dir.replace(".log", "_preds.png")), nrow=6, num_cls=self.cfg.num_cls, alpha=0.6)

    def save(self, nets):
        if self.best_epoch == self.epoch:
            for name, net in nets.items():
                torch.save(net, self.cfg.ckpt_path + f"_{name}.pth")
        logging.info("BEST : {:.3f}, EPOCH: {:3}".format(self.best_result, self.best_epoch + 1))
