import json
import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.pc_utils import read_ply


class ShapeCodesDataset(Dataset):
    def __init__(self, phase, config):
        super(ShapeCodesDataset, self).__init__()
        self.n_points = config.n_points
        self.data_root = config.data_root
        self.pc_root = config.pc_root
        self.path = config.split_path
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        with h5py.File(self.data_root, "r") as fp:
            self.zs = fp["{}_zs".format(phase)][:]
        self.noise = config.noise

    def __getitem__(self, index):
        data_id = self.all_data[index]
        pc_path = os.path.join(self.pc_root, data_id + ".ply")
        if not os.path.exists(pc_path):
            return self.__getitem__(index + 1)
        pc_n = read_ply(pc_path, with_normal=False)
        pc = pc_n[:, :3]
        normal = pc_n[:, -3:]
        sample_idx = random.sample(list(range(pc.shape[0])), self.n_points)
        pc = pc[sample_idx]
        normal = normal[sample_idx]
        normal = normal / (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-6)
        pc = pc + np.random.uniform(-self.noise, self.noise, (pc.shape[0], 1)) * normal
        pc = torch.tensor(pc, dtype=torch.float32)
        shape_code = torch.tensor(self.zs[index], dtype=torch.float32)
        return {"points": pc, "codes": shape_code, "id": data_id}

    def __len__(self):
        return len(self.zs)


def get_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == "train" if shuffle is None else shuffle

    dataset = ShapeCodesDataset(phase, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_shuffle,
        num_workers=config.num_workers,
    )
    return dataloader
