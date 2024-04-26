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

from model.pointcloudEncoder import PointNet2
from utils import read_ply, write_ply
from utils.file_utils import ensure_dir

sys.path.append("..")

from .base import BaseTrainer


class TrainerPcEncoder(BaseTrainer):
    def __init__(self, cfg):
        super(TrainerPcEncoder, self).__init__(cfg)
        self.build_net(cfg)
        self.losses_dict = {"train_loss": {}, "eval_loss": {}}
        
    def build_net(self, config):
        self.net = PointNet2().cuda()
        if len(config.gpu_ids) > 1:
            self.net = nn.DataParallel(self.net)

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
        
        print('********* Start Training ***********')

        for e in range(clock.epoch, nr_epochs):
            losses_train = []
            # begin iteration
            pbar = tqdm(train_loader)
            for b, data in enumerate(pbar):
                self.net.train()
                points = data["points"].cuda()
                codes = data["codes"].cuda()
                
                # train step
                pred = self.forward(points)
                loss = self.criterion(pred, codes)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses_train.append(loss.item())
                pbar.set_description(
                    "TRAIN - EPOCH[{}]-[{}] BATCH[{}]-[{}]".format(
                        e, nr_epochs, b, len(train_loader)
                    )
                )
                pbar.set_postfix(loss=loss.item())
                # validation step
                if clock.step % self.cfg.val_frequency == 0:
                    data = next(val_loader)
                    _, loss = self.evaluate(data)
                    self.losses_dict["eval_loss"][f"epoch {e}"] = loss.item()
                    pbar.set_description(
                        "EVAL - EPOCH[{}]-[{}] BATCH[{}]-[{}]".format(
                            e, nr_epochs, b, len(train_loader)
                        )
                    )
                    pbar.set_postfix(loss=loss.item())

                clock.tick()
                self.update_learning_rate()
            self.losses_dict["train_loss"][f"epoch {e}"] = sum(losses_train) / len(
                losses_train
            )
            clock.tock()

            if clock.epoch % self.cfg.save_frequency == 0:
                self.save_ckpt()

        # if clock.epoch % 10 == 0:
        self.save_ckpt("latest")

    def evaluate(self, data):
        with torch.no_grad:
            self.net.eval()
            points = data["points"].cuda()
            codes = data["codes"].cuda()
            pred = self.forward(points)
            loss = self.criterion(pred, codes)
            return pred, loss

    def encode_pointcloud(self, path):
        self.net.eval()
        if os.path.isfile(path):
            file_list = list(path if path.endswith(".ply") else None)
        elif os.path.isdir(path):
            file_list = [file for file in os.listdir(path) if file.endswith(".ply")]
        else:
            raise ValueError("Invalid path")

        # save_dir = os.path.join(cfg.exp_dir, "results/fake_z_ckpt{}_num{}_pc".format(args.ckpt, args.n_samples))
        save_dir = os.path.join(
            self.cfg.exp_dir,
            "results/pc2cad_{}".format(self.cfg.ckpt),
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        all_zs = []
        all_ids = []

        pbar = tqdm(file_list)
        for _, pc_path in enumerate(pbar):
            with torch.no_grad():
                # retrieve Data
                pc = read_ply(pc_path)
                sample_idx = random.sample(list(range(pc.shape[0])), self.cfg.n_points)
                pc = pc[sample_idx]
                pc = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).cuda()

                pred_z = self.forward(pc)
                pred_z = pred_z.detach().cpu().numpy()
                all_zs.append(pred_z)
                # print(pred_z.shape)
            all_ids.extend(pc_path)

            print(f'{pc_path.split("/")[-1]} encoded.')

        all_zs = np.concatenate(all_zs, axis=0)
        # save generated z
        save_path_zs = os.path.join(save_dir, datetime.datetime.now(), "all_zs.h5")
        ensure_dir(os.path.dirname(save_path_zs))
        with h5py.File(save_path_zs, "w") as fp:
            fp.create_dataset("zs", shape=all_zs.shape, data=all_zs)

        save_path_ids = os.path.join(
            save_dir, datetime.datetime.now(), "all_pc_ids.json"
        )
        with open(save_path_ids, "w") as fp:
            json.dump(all_ids, fp)

        return all_zs, save_path_zs

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(
                self.model_dir,
                "ckpt_epoch{}_num{}.pth".format(self.clock.epoch, self.cfg.n_points),
            )
            print("Saving checkpoint epoch {}...".format(self.clock.epoch))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

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
                "train_history": self.losses_dict,
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
            self.losses_dict = checkpoint["train_history"]
