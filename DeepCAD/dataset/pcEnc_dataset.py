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
        pc = read_ply(pc_path)
        sample_idx = random.sample(
            list(range(pc.shape[0])),
            self.n_points if self.n_points < pc.shape[0] else pc.shape[0],
        )
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
