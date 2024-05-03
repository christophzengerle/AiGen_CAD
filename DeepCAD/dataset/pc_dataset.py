import json
import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils.pc_utils import read_ply


class ShapeCodesDataset(Dataset):
    def __init__(self, phase, config, noise):
        super(ShapeCodesDataset, self).__init__()
        self.n_points = config.n_points
        self.z_path = config.z_path
        self.pc_root = config.pc_root
        self.path = config.split_path
        
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        with h5py.File(self.z_path, "r") as fp:
            self.zs = fp["{}_zs".format(phase)][:]
            
        self.noise = noise
        self.noiseAmount = config.noiseAmount
        
    def __getitem__(self, index):
        data_id = self.all_data[index]
        pc_path = os.path.join(self.pc_root, data_id + '.ply')
        if not os.path.exists(pc_path):
            return self.__getitem__(index + 1)
        pc = read_ply(pc_path)
        sample_idx = random.sample(list(range(pc.shape[0])), self.n_points)
        pc = pc[sample_idx]
        if self.noise:
            random_noise = True
            if random_noise:
                if random.choice([True, False]):
                    pc = pc + np.random.uniform(-self.noiseAmount, self.noiseAmount, (pc.shape[0], 1))
            else:
                pc = pc + np.random.uniform(-self.noiseAmount, self.noiseAmount, (pc.shape[0], 1))
        pc = torch.tensor(pc, dtype=torch.float32)
        shape_code = torch.tensor(self.zs[index], dtype=torch.float32)
        return {"points": pc, "codes": shape_code, "id": data_id}

    def __len__(self):
        return len(self.zs)


def get_dataloader(phase, config, noise=False, shuffle=None):
    is_shuffle = phase == "train" if shuffle is None else shuffle

    dataset = ShapeCodesDataset(phase, config, noise)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_shuffle,
        num_workers=config.num_workers,
    )
    return dataloader
