import logging
import os
import time
from os.path import join

import numpy as np
import torch
import yaml
from utils.utils_net import dividi_data


def initLogging(logFilename):
    """Init for logging"""
    logger = logging.getLogger("")

    if not logger.handlers:
        logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s-%(levelname)s]: %(message)s", datefmt="%y-%m-%d %H%M%S", filename=logFilename, filemode="w")
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s-%(levelname)s] %(message)s")
        console.setFormatter(formatter)
        logger.addHandler(console)


class Config(object):
    def __init__(self, args) -> None:
        super().__init__()
        # Initialize
        self.task_time = time.strftime("%Y-%m-%d_%H%M%S")
        self.config_file = args.c
        cfg = yaml.load(open(self.config_file), Loader=yaml.FullLoader)

        # Training Settings
        self.num_epoch = cfg["num_epochs"]
        self.num_workers = cfg["num_workers"]
        self.batch_size = cfg["batch_size"]
        self.lr = cfg["init_lr_rate"]

        # Device Settings
        self.fp16 = cfg["use_fp16"]
        if cfg["device"] != "cpu":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            self.fp16 = False

        # Task Settings
        self.task = cfg["task"]
        self.num_cls = cfg["num_cls"]
        self.pretrain = cfg["pretrain"]
        self.index = dividi_data()

        # Test?
        self.test = cfg["test"]

        # Make Folder
        taskfolder = f"./results/{self.task}"
        if not os.path.exists(taskfolder):
            os.makedirs(taskfolder)
        datafmt = "test" if self.test else self.task_time
        self.log_dir = f"{taskfolder}/{datafmt}.log"
        self.ckpt_path = f"{taskfolder}/{datafmt}"

        # logging information
        initLogging(self.log_dir)
        logging.info(f"Task: {self.task}")
        logging.info(f"Beginning_Time: {self.task_time}")
        logging.info(f"Batch_Size: {self.batch_size}, Learning_Rate: {self.lr}\n")
        logging.info(f"Config_File: {self.config_file}\n")

        return
