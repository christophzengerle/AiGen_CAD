import json
import math
import os
import random

import h5py
import numpy as np
import torch
import trimesh
from cadlib.macro import *
from torch.utils.data import DataLoader, Dataset
from trimesh import transformations
from utils.pc_utils import normalize_pc, read_ply


class ShapeCodesDataset(Dataset):
    def __init__(self, phase, config, noise):
        super(ShapeCodesDataset, self).__init__()
        self.n_points = config.n_points
        self.pc_root = config.pc_root
        self.raw_data = os.path.join(config.data_root, "cad_vec")  # h5 data root
        self.path = config.split_path
        self.max_total_len = config.max_total_len
        self.phase = phase

        # with open(self.path, "r") as fp:
        #     self.all_data = json.load(fp)[phase]

        # load all files for phase and remove faulty cad models from the dataset
        with open(self.path, "r") as data, open(
            config.faulty_cad_models_path, "r"
        ) as faulty:
            all_data_raw = json.load(data)[phase]

            # load list of faulty cad models to exclude from the dataset
            faulty_cad_models = json.load(faulty)
            if (
                len(faulty_cad_models[0].split("/")) > 2
                and faulty_cad_models[0].split(".")[1] == "obj"
            ):
                faulty_cad_models = [
                    "/".join(x.split("/")[2:]).split(".")[0] for x in faulty_cad_models
                ]

            self.all_data = [x for x in all_data_raw if x not in faulty_cad_models]

        self.noise = noise
        self.noiseAmount = config.noiseAmount

    def __getitem__(self, index):
        data_id = self.all_data[index]
        pc_path = os.path.join(self.pc_root, data_id + ".ply")
        if not os.path.exists(pc_path):
            return self.__getitem__(index + 1)

        if self.phase == "train":
            m = trimesh.load_mesh(pc_path)

            rotation = random.choice(np.arange(-180, 180, 45))
            elevation = random.choice(np.arange(-180, 180, 45))

            rotation_matrix = transformations.rotation_matrix(
                -1 * rotation * math.pi / 180, [0, 0, 1], [0, 0, 0]
            )

            m.apply_transform(rotation_matrix)

            elevation_matrix = transformations.rotation_matrix(
                -1 * elevation * math.pi / 180, [1, 0, 0], [0, 0, 0]
            )

            m.apply_transform(elevation_matrix)

            pc = m.vertices
            pc = normalize_pc(pc)

            if self.noise:
                if random.choice([True, False]):
                    pc = pc * (
                        np.random.uniform(-self.noiseAmount, self.noiseAmount, (1, 3))
                        + 1
                    )
                    pc = pc + np.random.uniform(
                        -self.noiseAmount / 10,
                        self.noiseAmount / 10,
                        (pc.shape[0], pc.shape[1]),
                    )

        else:
            pc = read_ply(pc_path)

        sample_idx = random.sample(
            list(range(pc.shape[0])),
            self.n_points if self.n_points < pc.shape[0] else pc.shape[0],
        )
        pc = pc[sample_idx]

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

        vecs = {"command": command, "args": args, "ids": data_id}

        return {"points": pc, "codes": vecs, "ids": data_id}

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
