import os
import sys

import h5py
import numpy as np
import torch
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.DataExchange import write_step_file
from tqdm import tqdm

from cadlib.macro import EOS_IDX
from config import ConfigAE
from dataset.cad_dataset import get_dataloader
from trainer import TrainerAE
from utils import ensure_dir

sys.path.append("..")
from cadlib.visualize import vec2CADsolid

# create experiment cfg containing all hyperparameters
cfg = ConfigAE()
print("data path:", cfg.data_root)

# create network and training agent
tr_agent = TrainerAE(cfg)


# define different modes
def reconstruct(cfg):
    # create dataloader
    test_loader = get_dataloader("test", cfg)
    print("Total number of test data:", len(test_loader))

    if cfg.outputs is None:
        cfg.outputs = "{}/results/test_{}".format(cfg.exp_dir, cfg.ckpt)
    ensure_dir(cfg.outputs)

    # evaluate
    pbar = tqdm(test_loader)
    for i, data in enumerate(pbar):
        batch_size = data["command"].shape[0]
        commands = data["command"]
        args = data["args"]
        gt_vec = (
            torch.cat([commands.unsqueeze(-1), args], dim=-1)
            .squeeze(1)
            .detach()
            .cpu()
            .numpy()
        )
        commands_ = gt_vec[:, :, 0]
        with torch.no_grad():
            outputs, _ = tr_agent.forward(data)
            batch_out_vec = tr_agent.logits2vec(outputs)

        for j in range(batch_size):
            out_vec = batch_out_vec[j]
            seq_len = commands_[j].tolist().index(EOS_IDX)

            data_id = data["id"][j].split("/")[-1]

            save_path = os.path.join(cfg.outputs, "{}_vec.h5".format(data_id))
            with h5py.File(save_path, "w") as fp:
                fp.create_dataset("out_vec", data=out_vec[:seq_len], dtype=np.int)
                fp.create_dataset("gt_vec", data=gt_vec[j][:seq_len], dtype=np.int)


def encode(cfg):
    # create dataloader
    save_dir = "{}/results".format(cfg.exp_dir)
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, "all_zs_ckpt{}.h5".format(cfg.ckpt))

    # fp = h5py.File(save_path, "w")
    fp = h5py.File("./data/all_zs.h5", "w")

    for phase in ["train", "validation", "test"]:
        train_loader = get_dataloader(phase, cfg, shuffle=False)

        # encode
        all_zs = []
        pbar = tqdm(train_loader)
        for i, data in enumerate(pbar):
            with torch.no_grad():
                z = tr_agent.encode(data, is_batch=True)
                z = z.detach().cpu().numpy()[:, 0, :]
                all_zs.append(z)
        all_zs = np.concatenate(all_zs, axis=0)
        print(all_zs.shape)
        fp.create_dataset("{}_zs".format(phase), data=all_zs)
    fp.close()


def decode(z_path, zs=None, batch_size=128, exportSTEP=False, checkBRep=False):
    if zs is None:
        with h5py.File(z_path, "r") as fp:
            zs = fp["zs"][:]
    save_dir = z_path.split(".")[0] + "_dec"
    ensure_dir(save_dir)

    # decode
    for i in range(0, len(zs), batch_size):
        with torch.no_grad():
            batch_z = torch.tensor(
                zs[i : i + batch_size], dtype=torch.float32
            ).unsqueeze(1)
            batch_z = batch_z.cuda()
            outputs = tr_agent.decode(batch_z)
            batch_out_vec = tr_agent.logits2vec(outputs)

        for j in range(len(batch_z)):
            out_vec = batch_out_vec[j]
            out_command = out_vec[:, 0]
            seq_len = out_command.tolist().index(EOS_IDX)
            out_vec = out_vec[:seq_len]

            save_path = os.path.join(save_dir, "{}.h5".format(i + j))
            with h5py.File(save_path, "w") as fp:
                fp.create_dataset("out_vec", data=out_vec, dtype=np.int32)

            if exportSTEP:
                try:
                    out_shape = vec2CADsolid(out_vec)

                except Exception:
                    print("load and create failed.")
                    continue

            if checkBRep:
                analyzer = BRepCheck_Analyzer(out_shape)
                if not analyzer.IsValid():
                    print("detect invalid.")
                    continue

            save_path = os.path.join(save_dir, "{}.step".format(i + j))
            write_step_file(out_shape, save_path)


# execute mode
if not cfg.test:

    # load from checkpoint if provided
    if cfg.cont:
        tr_agent.load_ckpt(cfg.ckpt)

    # create dataloader
    train_loader = get_dataloader("train", cfg)
    val_loader = get_dataloader("validation", cfg)
    val_loader = cycle(val_loader)

    tr_agent.train(train_loader, val_loader)


else:
    # load from checkpoint if provided
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    if cfg.mode == "rec":
        reconstruct(cfg)
    elif cfg.mode == "enc":
        encode(cfg)
    elif cfg.mode == "dec":
        decode(cfg.z_path, cfg.batch_size)
    else:
        raise ValueError
