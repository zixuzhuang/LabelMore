from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, Resize

from net.config import Config


class UNetDataset(Dataset):
    def __init__(self, data_type: str, cfg: Config):

        print(f"loading {data_type} data ...")
        self.cfg = cfg
        self.idxs = cfg.index[data_type]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        data = np.load(join('data/cache/slices', self.idxs[idx]),allow_pickle=True)
        image = torch.tensor(data['image'], dtype=torch.float32)
        label = torch.tensor(data['label'], dtype=torch.long)
        return image, label


def collate(samples):
    _data = UNetData()
    images, labels = map(list, zip(*samples))
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    _data.images = images[:, None, ...]
    _data.labels = labels
    return _data


def dataloader(cfg: Config):
    dataset = {}
    type_list = ["train", "valid"]
    for item in type_list:
        dataset[item] = DataLoader(
            UNetDataset(data_type=item, cfg=cfg),
            batch_size=cfg.batch_size,
            collate_fn=collate,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=True,
        )
    return dataset


class UNetData(object):
    def __init__(self) -> None:
        super().__init__()
        self.images = None
        self.labels = None
        return

    def to(self, device):
        self.images = self.images.to(device)
        self.labels = self.labels.to(device)
        return
