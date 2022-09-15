import argparse
import logging
import os
import time

# import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from net.config import Config
from net.dataloader import Transform, dataloader
from net.multi_losses import DiceLoss
from net.result import ResultSeg, cal_dice
from net.unet import UNet
from utils.utils_aug import augment
from utils.utils_img import show_seg
from utils.utils_net import dice_loss, get_args, lr_func, save_model

use_fp16 = True


def train(epoch):

    st = time.time()
    running_loss = 0.0
    this_lr = scheduler.get_last_lr()[0]
    net.train()
    for data in tqdm(dataset["train"], unit="batches", leave=False, dynamic_ncols=True):
        data.to(cfg.device)
        images, labels = augment(data.images, data.labels)
        images, labels = transfrom(images, labels)
        optimizer.zero_grad()
        with autocast(enabled=use_fp16):
            preds = net(images)
            loss = celoss(preds, labels) + dcloss(preds, labels) * 0.1
        if use_fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss = running_loss + loss.item()
    scheduler.step()

    ft = time.time()
    epoch_loss = running_loss / len(dataset["train"])
    logging.info("\n\nEPOCH: {}".format(epoch))
    logging.info("LOSS : {:.3f}, TIME: {:.1f}s, LR: {:.2e}".format(epoch_loss, ft - st, this_lr))
    return


@torch.no_grad()
def eval(dt):
    result = results[dt]
    result.init()
    net.eval()
    for data in dataset[dt]:
        data.to(cfg.device)
        with autocast(enabled=use_fp16):
            preds = net(data.images)
            dices = cal_dice(preds, data.labels, num_cls=cfg.num_cls)
        result.add_dice(dices)
        result.add_img(data.images, data.labels, preds)
    result.stastic()
    result.show()
    result.print(dt, epoch)
    return


if __name__ == "__main__":
    scaler = GradScaler()
    args = get_args(argparse.ArgumentParser())
    cfg = Config(args)
    transfrom = Transform(cfg.device)
    if cfg.pretrain == "None":
        net = UNet(1, cfg.num_cls, (64, 128, 128, 128, 128, 128, 128, 128)).to(cfg.device)
    else:
        net = torch.load(cfg.pretrain).to(cfg.device)
    eval_models = {}
    train_models = {cfg.task: net}
    models = train_models | eval_models
    celoss = nn.CrossEntropyLoss().to(cfg.device)
    dcloss = DiceLoss().to(cfg.device)
    optimizer = optim.Adam([{"params": items.parameters()} for keys, items in train_models.items()], lr=cfg.lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_func)
    dataset = dataloader(cfg)
    results = {"valid": ResultSeg(cfg), "test": ResultSeg(cfg)}

    for epoch in range(cfg.num_epoch):
        train(epoch + 1)
        eval("valid")
        results["valid"].save(models)
