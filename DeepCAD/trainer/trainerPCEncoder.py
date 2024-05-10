import datetime
import json
import os
import random
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append("..")
from model.pointcloudEncoder import PointNet2
from utils import read_ply
from utils.file_utils import walk_dir
from utils.step2png import transform

from .base import BaseTrainer


class TrainerPCEncoder(BaseTrainer):
    def __init__(self, cfg):
        super(TrainerPCEncoder, self).__init__(cfg)
        self.build_net(cfg)

    def build_net(self, config):
        self.net = PointNet2().cuda()
        # if len(config.gpu_ids) > 1:
        #     self.net = nn.DataParallel(self.net)

    def set_loss_function(self):
        self.criterion = nn.MSELoss().cuda()

    def set_optimizer(self, config):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), config.lr
        )  # , betas=(config.beta1, 0.9))
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, config.lr_step_size
        )

    def forward(self, data):
        data = data.cuda()
        pred = self.net(data)
        return pred

    def train(self, train_loader, val_loader):
        self.set_loss_function()
        self.set_optimizer(self.cfg)

        # start training
        clock = self.clock
        nr_epochs = self.cfg.nr_epochs

        print("********* Start Training ***********")

        for e in range(clock.epoch, nr_epochs):
            train_losses = []
            self.net.train()
            # begin iteration
            pbar = tqdm(train_loader)
            for b, data in enumerate(pbar):
                points = data["points"].cuda()
                codes = data["codes"].cuda()

                # train step
                pred = self.forward(points)
                loss = self.criterion(pred, codes)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

                pbar.set_description(
                    "TRAIN - EPOCH[{}]-[{}] BATCH[{}]-[{}]".format(
                        e, nr_epochs, b, len(train_loader)
                    )
                )
                pbar.set_postfix(loss=loss.item())

                clock.tick()

            self.record_losses(train_losses, mode="train")

            # validation step
            if clock.epoch % self.cfg.val_frequency == 0:
                eval_losses = []
                pbar = tqdm(val_loader)
                for i, data in enumerate(pbar):
                    _, loss = self.evaluate(data)

                    eval_losses.append(loss.item())

                    pbar.set_description(
                        "EVAL - EPOCH[{}]-[{}] BATCH[{}]-[{}]".format(
                            e, nr_epochs, i, len(val_loader)
                        )
                    )
                    pbar.set_postfix(loss=loss.item())

                self.record_losses(eval_losses, mode="eval")

            if clock.epoch % self.cfg.save_frequency == 0:
                self.save_ckpt()

            self.record_and_update_learning_rate()

            clock.tock()

        # if clock.epoch % 10 == 0:
        self.save_ckpt("latest")

    def evaluate(self, data):
        with torch.no_grad():
            self.net.eval()
            points = data["points"].cuda()
            codes = data["codes"].cuda()
            pred = self.forward(points)
            loss = self.criterion(pred, codes)
        return pred, loss

    def test(self, test_loader):
        raise NotImplementedError
        # """evaluatinon during training"""

    def encode_pointcloud(self, path):
        self.net.eval()
        if os.path.isfile(path):
            if path.endswith(".ply"):
                file_list = [path]
                save_name = path.split("/")[-1].split(".")[0]
            else:
                raise ValueError("Invalid file format")
        elif os.path.isdir(path):
            file_list = [file for file in walk_dir(path) if file.endswith(".ply")]
            save_name = os.path.basename(os.path.normpath(path))
        else:
            raise ValueError("Invalid path")

        out_dir = os.path.join(
            self.cfg.exp_dir,
            "results/pc2cad/",
            self.cfg.ckpt,
            save_name
            + "_"
            + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
        )

        if self.cfg.output is not None:
            if os.path.isfile(self.cfg.output):
                out_dir = os.path.split(self.cfg.output)[0]
            elif os.path.isdir(self.cfg.output):
                out_dir = self.cfg.output
            else:
                try:
                    os.makedirs(self.cfg.output)
                    out_dir = self.cfg.output
                except Exception as e:
                    print("Output-path is invalid. Using default path.")

        save_dir = out_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        all_zs = {"zs": [], "z_paths": [], "pc_paths": []}

        print(f"File-List: {file_list}")
        if len(file_list) > 0:
            pbar = tqdm(file_list)
            for i, pc_path in enumerate(pbar):
                with torch.no_grad():
                    # retrieve Data
                    pc = read_ply(pc_path)
                    try:
                        sample_idx = random.sample(
                            list(range(pc.shape[0])),
                            (
                                self.cfg.n_points
                                if self.cfg.n_points < pc.shape[0]
                                else pc.shape[0]
                            ),
                        )
                    except ValueError:
                        print(
                            f"Point cloud {pc_path.split('/')[-1]} has less than {self.cfg.n_points} points."
                        )
                        continue
                    pc = pc[sample_idx]
                    pc = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).cuda()

                    pred_z = self.forward(pc)
                    pred_z = pred_z.detach().cpu().numpy()

                if len(pred_z) > 0:
                    # save generated z
                    file_name = pc_path.split("/")[-1].split(".")[0]
                    save_path_zs = os.path.join(save_dir, f"{file_name}.h5")

                    all_zs["zs"].append(pred_z)
                    all_zs["z_paths"].append(save_path_zs)
                    all_zs["pc_paths"].append(pc_path)

                    with h5py.File(save_path_zs, "w") as fp:
                        fp.create_dataset("zs", shape=pred_z.shape, data=pred_z)

                    if self.cfg.expSourcePNG:
                        transform(
                            pc_path,
                            save_path_zs.split(".")[0],
                            135,
                            45,
                            "medium",
                            exp_png=self.cfg.expSourcePNG,
                            make_gif=False,
                        )

                    print(
                        f'{pc_path.split("/")[-1]} encoded and saved to: {save_path_zs}'
                    )

                else:
                    raise print(f"No Data to Encode for file {pc_path}.")

            save_path_ids = os.path.join(save_dir, "all_pc_ids.json")
            with open(save_path_ids, "w") as fp:
                json.dump({"Point-Cloud paths:": all_zs["pc_paths"]}, fp)

            print("********* PointCloud Encoding Completed ***********")

            return all_zs

        else:
            raise ValueError("No .ply files found in the provided path.")

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(
                self.model_dir,
                "ckpt_epoch{}_num{}.pth".format(self.clock.epoch, self.cfg.n_points),
            )
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

        print("Saving checkpoint epoch {}...".format(self.clock.epoch))
        if isinstance(self.net, nn.DataParallel):
            model_state_dict = self.net.module.cpu().state_dict()
        else:
            model_state_dict = self.net.cpu().state_dict()

        torch.save(
            {
                "clock": self.clock.make_checkpoint(),
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            save_path,
        )

        self.net.cuda()

    def load_ckpt(self, name=None):
        """load checkpoint from saved checkpoint"""
        # name = (name if name == "latest" else "ckpt_epoch{}".format(name))
        load_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading checkpoint from {} ...".format(load_path))
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.net.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.clock.restore_checkpoint(checkpoint["clock"])

    def record_and_update_learning_rate(self):
        """record and update learning rate"""
        self.train_tb.add_scalar(
            "learning_rate", self.optimizer.param_groups[-1]["lr"], self.clock.epoch
        )
        self.scheduler.step()

    def record_losses(self, losses, mode="train"):
        """record loss to tensorboard"""
        tb = self.train_tb if mode == "train" else self.val_tb
        tb.add_scalar("Loss", np.sum(losses) / len(losses), self.clock.epoch)
