import json
import os
import random
import math

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils.pc_utils import normalize_pc, read_ply

import trimesh
from trimesh import transformations


class ShapeCodesDataset(Dataset):
    def __init__(self, phase, config, noise):
        super(ShapeCodesDataset, self).__init__()
        self.n_points = config.n_points
        self.z_path = config.z_path
        self.pc_root = config.pc_root
        self.path = config.split_path
        self.phase = phase

        # with open(self.path, "r") as fp:
        #     self.all_data = json.load(fp)[phase]
        
                
        # with h5py.File(self.z_path, "r") as fp:
        #     self.zs = fp["{}_zs".format(phase)][:]
        

        # load all files for phase and remove faulty cad models from the dataset
        with open(self.path, "r") as json_data, h5py.File(self.z_path, "r") as vec_data,  open(
            config.faulty_cad_models_path, "r"
        ) as faulty:
            json_data_raw = json.load(json_data)[phase]
            vec_data_raw = vec_data["{}_zs".format(phase)][:]

            # load list of faulty cad models to exclude from the dataset
            faulty_cad_models = json.load(faulty)
            if (
                len(faulty_cad_models[0].split("/")) > 2
                and faulty_cad_models[0].split(".")[1] == "obj"
            ):
                faulty_cad_models = [
                    "/".join(x.split("/")[2:]).split(".")[0] for x in faulty_cad_models
                ]

                faulty_cad_models_indices = np.where(np.in1d(json_data_raw, faulty_cad_models))[0]
                
                       
            self.all_data = [x for x in json_data_raw if x not in faulty_cad_models]
            self.zs = [x for idx, x in enumerate(vec_data_raw) if idx not in faulty_cad_models_indices]
            
            

        self.noise = noise
        self.noiseAmount = config.noiseAmount

    def __getitem__(self, index):
        data_id = self.all_data[index]
        pc_path = os.path.join(self.pc_root, data_id + ".ply")
        if not os.path.exists(pc_path):
            return self.__getitem__(index + 1)
        
        
        if self.phase == "train":
            # read point cloud and apply random rotation and elevation
            m = trimesh.load_mesh(pc_path)

            rotation = random.choice(np.arange(-90, 101.25, 11.25))
            elevation = random.choice(np.arange(-90, 101.25, 11.25))

            rotation_matrix = transformations.rotation_matrix(
                -1 * rotation * math.pi / 180, [0, 0, 1], [0, 0, 0]
            )

            m.apply_transform(rotation_matrix)

            elevation_matrix = transformations.rotation_matrix(
                -1 * elevation * math.pi / 180, [1, 0, 0], [0, 0, 0]
            )

            m.apply_transform(elevation_matrix)

            pc = m.vertices

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
            pc = normalize_pc(pc)

        else:
            pc = read_ply(pc_path)

        sample_idx = random.sample(
            list(range(pc.shape[0])),
            self.n_points if self.n_points < pc.shape[0] else pc.shape[0],
        )
        pc = pc[sample_idx]

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
