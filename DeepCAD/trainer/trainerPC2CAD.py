from collections import OrderedDict
import datetime
import json
import os
import random
import sys

from .loss import CADLoss
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from tqdm import tqdm

from utils.step_utils import create_step_file, step_file_exists

sys.path.append("..")
from cadlib.macro import *
from model.pointcloud2cad import PointCloud2CAD
from .trainerAE import TrainerAE
from .trainerPCEncoder import TrainerPCEncoder
from utils import read_ply
from utils.file_utils import walk_dir
from utils.step2png import transform

from cadlib.visualize import vec2CADsolid
from trainer.scheduler import GradualWarmupScheduler

from .base import BaseTrainer


class TrainerPC2CAD(BaseTrainer):
    def __init__(self, cfg):
        super(TrainerPC2CAD, self).__init__(cfg)
        
        self.build_net(cfg)

    def build_net(self, cfg):
        self.trainer_pc_enc = TrainerPCEncoder(cfg)
        self.trainer_ae = TrainerAE(cfg)
        self.net = PointCloud2CAD(self.trainer_pc_enc, self.trainer_ae)
        # if len(cfg.gpu_ids) > 1:
        #     self.net = nn.DataParallel(self.net)

    def set_loss_function(self):
        self.criterion = CADLoss(self.cfg).cuda()

    def set_optimizer(self, cfg):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), cfg.lr
        )  # , betas=(config.beta1, 0.9))
        # self.scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer, config.lr_step_size
        # )
        self.scheduler = GradualWarmupScheduler(
            self.optimizer, 1.0, cfg.warmup_step
        )

    def forward(self, data):
        data = data.cuda()
        pred = self.net(data)
        return pred

    def train(self, train_loader, val_loader, test_loader):
        self.set_loss_function()
        self.set_optimizer(self.cfg)

        # start training
        clock = self.clock
        nr_epochs = self.cfg.nr_epochs

        print("********* Start Training ***********")

        for e in range(clock.epoch, nr_epochs):
            train_losses = {"losses_cmd": [], "losses_args": []}
            self.net.train()
            # begin iteration
            pbar = tqdm(train_loader)
            for b, data in enumerate(pbar):
                points = data["points"]
                codes = data["codes"]

                # train step
                pred = self.forward(points)
                loss = self.criterion(pred, codes)
                self.update_network(loss)

                train_losses["losses_cmd"].append(loss["loss_cmd"].item())
                train_losses["losses_args"].append(loss["loss_args"].item())

                pbar.set_description(
                    "TRAIN - EPOCH[{}]-[{}] BATCH[{}]-[{}]".format(
                        e, nr_epochs, b, len(train_loader)
                    )
                )
                pbar.set_postfix(OrderedDict({k: v.item() for k, v in loss.items()}))

                clock.tick()

            self.record_losses(train_losses, mode="train")

            # validation step
            if clock.epoch % self.cfg.val_frequency == 0:
                eval_losses = {"losses_cmd": [], "losses_args": []}
                pbar = tqdm(val_loader)
                for i, data in enumerate(pbar):
                    _, loss = self.evaluate(data)

                    eval_losses["losses_cmd"].append(loss["loss_cmd"].item())
                    eval_losses["losses_args"].append(loss["loss_args"].item())

                    pbar.set_description(
                        "EVAL - EPOCH[{}]-[{}] BATCH[{}]-[{}]".format(
                            e, nr_epochs, i, len(val_loader)
                        )
                    )
                    pbar.set_postfix(OrderedDict({k: v.item() for k, v in loss.items()}))

                self.record_losses(eval_losses, mode="eval")

            if clock.epoch % self.cfg.save_frequency == 0:
                self.test(test_loader)
                self.save_ckpt()

            self.record_and_update_learning_rate()

            clock.tock()

        # if clock.epoch % 10 == 0:
        self.save_ckpt("latest")

    def evaluate(self, data):
        with torch.no_grad():
            self.net.eval()
            points = data["points"]
            codes = data["codes"]
            pred = self.forward(points)
            loss = self.criterion(pred, codes)
        return pred, loss

    def test(self, test_loader):
        """evaluatinon during training"""
        self.net.eval()
        pbar = tqdm(test_loader)

        test_losses = {"losses_cmd": [], "losses_args": []}
        all_ext_args_comp = []
        all_line_args_comp = []
        all_arc_args_comp = []
        all_circle_args_comp = []

        for i, data in enumerate(pbar):
            with torch.no_grad():
                points = data["points"]
                codes = data["codes"]

                pred = self.forward(points)
                out_args = (
                    torch.argmax(torch.softmax(pred["args_logits"], dim=-1), dim=-1) - 1
                )
                out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)

            loss = self.criterion(pred, codes)
            test_losses["losses_cmd"].append(loss["loss_cmd"].item())
            test_losses["losses_args"].append(loss["loss_args"].item())
            
            pbar.set_description(
                        "TEST - EPOCH[{}]-[{}] BATCH[{}]-[{}]".format(
                            self.clock.epoch, self.nr_epochs, i, len(test_loader)
                        )
                    )
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in loss.items()}))


            commands, args = codes["command"].cuda(), codes["args"].cuda()

            gt_commands = commands.squeeze(1).long().detach().cpu().numpy()  # (N, S)
            gt_args = args.squeeze(1).long().detach().cpu().numpy()  # (N, S, n_args)

            ext_pos = np.where(gt_commands == EXT_IDX)
            line_pos = np.where(gt_commands == LINE_IDX)
            arc_pos = np.where(gt_commands == ARC_IDX)
            circle_pos = np.where(gt_commands == CIRCLE_IDX)

            args_comp = (gt_args == out_args).astype(np.int32)
            all_ext_args_comp.append(args_comp[ext_pos][:, -N_ARGS_EXT:])
            all_line_args_comp.append(args_comp[line_pos][:, :2])
            all_arc_args_comp.append(args_comp[arc_pos][:, :4])
            all_circle_args_comp.append(args_comp[circle_pos][:, [0, 1, 4]])

        self.record_losses(test_losses, mode="test")

        all_ext_args_comp = np.concatenate(all_ext_args_comp, axis=0)
        sket_plane_acc = np.mean(all_ext_args_comp[:, :N_ARGS_PLANE])
        sket_trans_acc = np.mean(
            all_ext_args_comp[:, N_ARGS_PLANE : N_ARGS_PLANE + N_ARGS_TRANS]
        )
        extent_one_acc = np.mean(all_ext_args_comp[:, -N_ARGS_EXT_PARAM])
        line_acc = np.mean(np.concatenate(all_line_args_comp, axis=0))
        arc_acc = np.mean(np.concatenate(all_arc_args_comp, axis=0))
        circle_acc = np.mean(np.concatenate(all_circle_args_comp, axis=0))

        self.test_tb.add_scalars(
            "args_acc",
            {
                "line": line_acc,
                "arc": arc_acc,
                "circle": circle_acc,
                "plane": sket_plane_acc,
                "trans": sket_trans_acc,
                "extent": extent_one_acc,
            },
            global_step=self.clock.epoch,
        )
        
    def test_model_acc(self, test_loader):
        save_dir = os.path.join(
            self.cfg.exp_dir,
            "evaluation/accuracy/",
            self.cfg.ckpt,
            str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        TOLERANCE = 3

        # overall accuracy
        avg_cmd_acc = [] # ACC_cmd
        avg_param_acc = [] # ACC_param

        # accuracy w.r.t. each command type
        each_cmd_cnt = np.zeros((len(ALL_COMMANDS),))
        each_cmd_acc = np.zeros((len(ALL_COMMANDS),))

        # accuracy w.r.t each parameter
        args_mask = CMD_ARGS_MASK.astype(np.float32)
        N_ARGS = args_mask.shape[1]
        each_param_cnt = np.zeros([*args_mask.shape])
        each_param_acc = np.zeros([*args_mask.shape])        

        pbar = tqdm(test_loader)

        for batch_nr, data in enumerate(pbar):
            with torch.no_grad():
                batch_cmd_acc = [] # ACC_cmd
                batch_param_acc = [] # ACC_param
                
                points = data["points"]
                codes = data["codes"]
                
                gt_cmd = codes["command"].detach().cpu().numpy()
                gt_param = codes["args"].detach().cpu().numpy()

                pred = self.forward(points)
                batch_out_vec = self.trainer_ae.logits2vec(pred)
                
                out_cmd = batch_out_vec[:, :, 0]
                out_param = batch_out_vec[:, :, 1:]
                
                cmd_acc = (out_cmd == gt_cmd).astype(np.int32)
                
            for i in range(len(data)):
                param_acc = []
                for j in range(len(gt_cmd[i])):
                    cmd = gt_cmd[i][j]
                    each_cmd_cnt[cmd] += 1
                    each_cmd_acc[cmd] += cmd_acc[i][j]
                    if cmd in [SOL_IDX, EOS_IDX]:
                        continue

                    if out_cmd[i][j] == gt_cmd[i][j]: # NOTE: only account param acc for correct cmd
                        tole_acc = (np.abs(out_param[i][j] - gt_param[i][j]) < TOLERANCE).astype(np.int32)
                        # filter param that do not need tolerance (i.e. requires strictly equal)
                        if cmd == EXT_IDX:
                            tole_acc[-2:] = (out_param[i][j] == gt_param[i][j]).astype(np.int32)[-2:]
                        elif cmd == ARC_IDX:
                            tole_acc[3] = (out_param[i][j] == gt_param[i][j]).astype(np.int32)[3]

                        valid_param_acc = tole_acc[args_mask[cmd].astype(np.bool_)].tolist()
                        param_acc.extend(valid_param_acc)

                        each_param_cnt[cmd, np.arange(N_ARGS)] += 1
                        each_param_acc[cmd, np.arange(N_ARGS)] += tole_acc


                mean_param_acc = np.mean(param_acc) if len(param_acc) > 0 else 0
                avg_param_acc.append(mean_param_acc)
                batch_param_acc.append(mean_param_acc)
                
                mean_cmd_acc = np.mean(cmd_acc[i])
                avg_cmd_acc.append(mean_cmd_acc)
                batch_cmd_acc.append(mean_cmd_acc)          
                
                
            pbar.set_description(
                    "TEST ACCURACY - BATCH[{}]-[{}]".format(
                        batch_nr, len(test_loader)
                    )
                )
            pbar.set_postfix(OrderedDict({"CMD-Acc": np.mean(batch_cmd_acc), "Args-Acc" : np.mean(batch_param_acc)}))

        save_path = os.path.join(save_dir, "test_acc_stats.txt")
        fp = open(save_path, "w")
        # overall accuracy (averaged over all data)
        avg_cmd_acc = np.mean(avg_cmd_acc)
        print("avg command acc (ACC_cmd):", avg_cmd_acc, file=fp)
        
        print(avg_param_acc)
        print(np.mean(avg_param_acc))
        avg_param_acc = np.mean(avg_param_acc)
        print("avg param acc (ACC_param):", avg_param_acc, file=fp)

        # acc of each command type
        each_cmd_acc = each_cmd_acc / (each_cmd_cnt + 1e-6)
        print("each command count:", each_cmd_cnt, file=fp)
        print("each command acc:", each_cmd_acc, file=fp)

        # acc of each parameter type
        each_param_acc = each_param_acc * args_mask
        each_param_cnt = each_param_cnt * args_mask
        each_param_acc = each_param_acc / (each_param_cnt + 1e-6)
        for i in range(each_param_acc.shape[0]):
            print(ALL_COMMANDS[i] + " param acc:", each_param_acc[i][args_mask[i].astype(np.bool_)], file=fp)
        fp.close()

        with open(save_path, "r") as fp:
            res = fp.readlines()
            for l in res:
                print(l, end='')
                
                
    def test_model_chamfer_dist():
        pass
        

    def pc2cad(self):
        self.net.eval()
        path = self.cfg.pc_root
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

        # save_dir = os.path.join(cfg.exp_dir, "results/fake_z_ckpt{}_num{}_pc".format(args.ckpt, args.n_samples))
        save_dir = os.path.join(
            self.cfg.exp_dir,
            "results/pc2cad/",
            self.cfg.ckpt,
            save_name
            + "_"
            + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print(f"File-List: {file_list}")
        if len(file_list) > 0:
            pbar = tqdm(file_list)
            for i, pc_path in enumerate(pbar):
                with torch.no_grad():
                    # retrieve Data
                    pc = read_ply(pc_path)
                    try:
                        sample_idx = random.sample(
                            list(range(pc.shape[0])), self.cfg.n_points
                        )
                    except ValueError:
                        print(
                            f"Point cloud {pc_path.split('/')[-1]} has less than {self.cfg.n_points} points."
                        )
                        continue
                    pc = pc[sample_idx]
                    pc = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).cuda()

                    pred = self.forward(pc)
                    batch_out_vec = self.trainer_ae.logits2vec(pred)
                    out_vec = batch_out_vec.squeeze(0)

                if len(out_vec) > 0:
                    # save generated z
                    file_name = pc_path.split("/")[-1].split(".")[0]
                    save_path = os.path.join(save_dir, f"{file_name}")
                    if self.cfg.expSourcePNG:
                        transform(
                            pc_path,
                            save_path,
                            135,
                            45,
                            "medium",
                            exp_png=self.cfg.expSourcePNG,
                            make_gif=False,
                        )

                        print(
                            f'{pc_path.split("/")[-1]} encoded and saved to: {save_path}'
                        )

                    out_command = out_vec[:, 0]
                    seq_len = out_command.tolist().index(EOS_IDX)
                    out_vec = out_vec[:seq_len]

                    # check generated CAD-Shape and save it
                    out_shape = None
                    is_valid_BRep = False
                    try:
                        out_shape = vec2CADsolid(out_vec)
                        is_valid_BRep = True
                    except Exception as e:
                        print(
                            f"Creation of CAD-Solid for file {save_path} failed.\n"
                            + str(e.with_traceback)
                        )

                        # check generated CAD-Shape
                        # if invalid -> generate again
                        for cnt_retries in range(1, self.cfg.n_checkBrep_retries + 1):
                            print(
                                f"Trying to create a new CAD-Solid. Attempt {cnt_retries}/{self.cfg.n_checkBrep_retries}"
                            )

                            new_batch_output = self.forward(pc)
                            out_batch_vec = self.trainer_ae.logits2vec(new_batch_output)
                            out_vec = out_batch_vec.squeeze(0)

                            out_command = out_vec[:, 0]
                            seq_len = out_command.tolist().index(EOS_IDX)
                            out_vec = out_vec[:seq_len]

                            try:
                                out_shape = vec2CADsolid(out_vec)
                                analyzer = BRepCheck_Analyzer(out_shape)
                                if analyzer.IsValid():
                                    print("Valid BRep-Model detected.")
                                    is_valid_BRep = True
                                    break
                                else:
                                    print("invalid BRep-Model detected.")
                                    continue

                            except Exception as e:
                                print(
                                    f"Creation of CAD-Solid for file {save_path} failed.\n"
                                    + str(e.with_traceback)
                                )
                                continue

                    if not is_valid_BRep:
                        print("Could not create valid BRep-Model!")
                        continue

                    save_path_vec = save_path + "_dec.h5"
                    with h5py.File(save_path_vec, "w") as fp:
                        fp.create_dataset("out_vec", data=out_vec, dtype=np.int32)

                    step_save_path = save_path + "_dec.step"
                    if self.cfg.expSTEP:
                        try:
                            create_step_file(out_shape, step_save_path)
                        except Exception as e:
                            print(str(e.with_traceback))
                            continue

                    if self.cfg.expPNG or self.cfg.expGIF:
                        try:
                            png_path = step_save_path.split(".")[0]
                            if step_file_exists(step_save_path):
                                transform(
                                    step_save_path,
                                    png_path,
                                    135,
                                    45,
                                    "medium",
                                    exp_png=self.cfg.expPNG,
                                    make_gif=self.cfg.expGIF,
                                )
                                print(f"Image-Output for {png_path} created.")
                            else:
                                print(
                                    f'no .STEP-File for {step_save_path.split("/")[-1]} found.\nTrying to create .STEP-File'
                                )
                                try:
                                    create_step_file(out_shape, step_save_path)
                                except Exception as e:
                                    raise Exception(str(e.with_traceback))
                        except Exception as e:
                            print(
                                f"Creation of Image-Output for {save_path.split('/')[-1]} failed.\n"
                                + str(e.with_traceback)
                            )
                            continue

                else:
                    raise print(f"No Data to Encode for file {pc_path}.")

        else:
            raise ValueError("No .ply files found in the provided path.")

        print("********* Prediction of CAD-Model from PointCloud Completed ***********")

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
        if mode == "train":
            tb = self.train_tb
        elif mode == "eval":
            tb = self.val_tb
        elif mode == "test":
            tb = self.test_tb
        else:
            raise ValueError("tensorboard-mode should be train, eval or test")
        tb.add_scalar(
            "Loss_CMD",
            np.sum(losses["losses_cmd"]) / len(losses["losses_cmd"]),
            self.clock.epoch,
        )
        tb.add_scalar(
            "Loss_ARGS",
            np.sum(losses["losses_args"]) / len(losses["losses_args"]),
            self.clock.epoch,
        )
