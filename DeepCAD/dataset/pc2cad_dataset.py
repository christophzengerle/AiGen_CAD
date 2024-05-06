import json
import os
import random

import h5py
import numpy as np
import torch
from cadlib.macro import *
from torch.utils.data import DataLoader, Dataset
from utils.pc_utils import read_ply


class ShapeCodesDataset(Dataset):
    def __init__(self, phase, config, noise):
        super(ShapeCodesDataset, self).__init__()
        self.n_points = config.n_points
        self.pc_root = config.pc_root
        self.raw_data = os.path.join(config.data_root, "cad_vec")  # h5 data root
        self.path = config.split_path
        self.max_total_len = config.max_total_len

        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        self.noise = noise
        self.noiseAmount = config.noiseAmount

    def __getitem__(self, index):
        data_id = self.all_data[index]
        pc_path = os.path.join(self.pc_root, data_id + ".ply")
        if not os.path.exists(pc_path):
            return self.__getitem__(index + 1)
        pc = read_ply(pc_path)
        sample_idx = random.sample(list(range(pc.shape[0])), self.n_points)
        pc = pc[sample_idx]

        # Noise
        if self.noise:
            random_noise = True
            if random_noise:
                if random.choice([True, False]):
                    pc = pc + np.random.uniform(
                        -self.noiseAmount, self.noiseAmount, (pc.shape[0], 1)
                    )
            else:
                pc = pc + np.random.uniform(
                    -self.noiseAmount, self.noiseAmount, (pc.shape[0], 1)
                )

            # random_n_points = True

            # if random_n_points:
            #     n_points = random.choice([512, 1024, 2048, 4096])
            #     sample_idx = random.sample(list(range(pc.shape[0])), n_points)
            #     pc = pc[sample_idx]

            # else:
            #     sample_idx = random.sample(list(range(pc.shape[0])), self.n_points)
            #     pc = pc[sample_idx]

        pc = torch.tensor(pc, dtype=torch.float32)

        vec_path = os.path.join(self.raw_data, data_id + ".h5")
        with h5py.File(vec_path, "r") as fp:
            cad_vec = fp["vec"][:]  # (len, 1 + N_ARGS)

        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate(
            [cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0
        )
        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)

        vecs = {"command": command, "args": args, "id": data_id}

        return {"points": pc, "codes": vecs, "id": data_id}

    def __len__(self):
        return len(self.all_data)


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
