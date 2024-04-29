import os
import sys

import h5py
import numpy as np
import torch
from cadlib.macro import EOS_IDX
from config import ConfigAE
from dataset.cad_dataset import get_dataloader
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.DataExchange import write_step_file
from tqdm import tqdm
from trainer import TrainerAE
from utils import ensure_dir

sys.path.append("..")
from cadlib.visualize import vec2CADsolid
from utils import cycle
from utils.cad2cad_utils import decode, encode, reconstruct


def decode_pc_zs(pc_config):
    sys.argv[1:] = list(
        map(
            str,
            [
                "--exec",
                pc_config.exec,
                "--exp_name",
                pc_config.ae_exp_name,
                "--ckpt",
                pc_config.ae_ckpt,
                "--gpu_ids",
                pc_config.gpu_ids,
            ],
        )
    )
    cfg = ConfigAE()
    cfg.set_pc_decoder_configuration(pc_config)
    tr_agent = TrainerAE(cfg)
    tr_agent.load_ckpt(cfg.ckpt)
    decode(cfg, tr_agent)


def main():
    # create experiment cfg containing all hyperparameters
    cfg = ConfigAE()
    print("data path:", cfg.data_root)

    # create network and training agent
    tr_agent = TrainerAE(cfg)

    # execute mode
    if cfg.exec == "train":

        # load from checkpoint if provided
        if cfg.cont:
            tr_agent.load_ckpt(cfg.ckpt)

        # create dataloader
        train_loader = get_dataloader("train", cfg)
        val_loader = get_dataloader("validation", cfg)
        val_loader = cycle(val_loader)

        tr_agent.train(train_loader, val_loader)

    elif cfg.exec == "test":
        # load from checkpoint if provided
        tr_agent.load_ckpt(cfg.ckpt)
        tr_agent.net.eval()

        if cfg.mode == "rec":
            reconstruct(cfg, tr_agent)
        elif cfg.mode == "enc":
            encode(cfg, tr_agent)
        elif cfg.mode == "dec":
            decode(cfg, tr_agent)
        else:
            raise ValueError("Invalid mode.")

    else:
        raise ValueError(
            "Invalid execution type. Please specify --exec 'train' or 'test' mode"
        )


if __name__ == "__main__":
    main()
