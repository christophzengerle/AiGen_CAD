import datetime
import os
import sys
from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from tqdm import tqdm

from .base import BaseTrainer
from .loss import CADLoss
from .scheduler import GradualWarmupScheduler

sys.path.append("..")
from cadlib.macro import trim_vec_EOS
from cadlib.macro import *
from model import CADTransformer
from cadlib.visualize import vec2CADsolid
from utils.file_utils import walk_dir
from utils.step2png import transform
from utils.step_utils import create_step_file, step_file_exists


class TrainerAE(BaseTrainer):
    def __init__(self, cfg):
        super(TrainerAE, self).__init__(cfg)
        self.build_net(cfg)

    def build_net(self, cfg):
        self.net = CADTransformer(cfg).cuda()
        if len(cfg.gpu_ids) > 1:
            self.net = nn.DataParallel(self.net)

    def set_optimizer(self, cfg):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = optim.Adam(self.net.parameters(), cfg.lr)
        self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, cfg.warmup_step)

    def set_loss_function(self):
        self.loss_func = CADLoss(self.cfg).cuda()

    def forward(self, data):
        commands = data["command"].cuda()  # (N, S)
        args = data["args"].cuda()  # (N, S, N_ARGS)

        outputs = self.net(commands, args)
        loss_dict = self.loss_func(outputs, data)

        return outputs, loss_dict

    def encode(self, data, is_batch=False):
        """encode into latent vectors"""
        commands = data["command"].cuda()
        args = data["args"].cuda()
        if not is_batch:
            commands = commands.unsqueeze(0)
            args = args.unsqueeze(0)
        z = self.net(commands, args, encode_mode=True)
        return z

    def decode(self, z):
        """decode given latent vectors"""
        outputs = self.net(None, None, z=z)
        return outputs

    def logits2vec(self, outputs, refill_pad=True, to_numpy=True):
        """network outputs (logits) to final CAD vector"""
        out_command = torch.argmax(
            torch.softmax(outputs["command_logits"], dim=-1), dim=-1
        )  # (N, S)
        out_args = (
            torch.argmax(torch.softmax(outputs["args_logits"], dim=-1), dim=-1) - 1
        )  # (N, S, N_ARGS)
        if refill_pad:  # fill all unused element to -1
            mask = ~torch.tensor(CMD_ARGS_MASK).bool().cuda()[out_command.long()]
            out_args[mask] = -1

        out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
        if to_numpy:
            out_cad_vec = out_cad_vec.detach().cpu().numpy()
        return out_cad_vec

    def train(self, train_loader, val_loader):
        self.set_loss_function()
        self.set_optimizer(self.cfg)
        # start training
        # start training
        clock = self.clock

        for e in range(clock.epoch, self.cfg.nr_epochs):
            # begin iteration
            pbar = tqdm(train_loader)
            for b, data in enumerate(pbar):
                # train step
                outputs, losses = self.train_func(data)

                pbar.set_description("EPOCH[{}][{}]".format(e, b))
                pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

                # validation step
                if clock.step % self.cfg.val_frequency == 0:
                    data = next(val_loader)
                    outputs, losses = self.val_func(data)

                clock.tick()

                self.update_learning_rate()

            clock.tock()

            if clock.epoch % self.cfg.save_frequency == 0:
                self.save_ckpt()

            # if clock.epoch % 10 == 0:
            self.save_ckpt("latest")

    def test(self, test_loader):
        """evaluatinon during training"""
        self.net.eval()
        pbar = tqdm(test_loader)
        pbar.set_description("EVALUATE[{}]".format(self.clock.epoch))

        all_ext_args_comp = []
        all_line_args_comp = []
        all_arc_args_comp = []
        all_circle_args_comp = []

        for i, data in enumerate(pbar):
            with torch.no_grad():
                commands = data["command"].cuda()
                args = data["args"].cuda()
                outputs = self.net(commands, args)
                out_args = (
                    torch.argmax(torch.softmax(outputs["args_logits"], dim=-1), dim=-1)
                    - 1
                )
                out_args = out_args.long().detach().cpu().numpy()  # (N, S, n_args)

            gt_commands = commands.squeeze(1).long().detach().cpu().numpy()  # (N, S)
            gt_args = args.squeeze(1).long().detach().cpu().numpy()  # (N, S, n_args)

            ext_pos = np.where(gt_commands == EXT_IDX)
            line_pos = np.where(gt_commands == LINE_IDX)
            arc_pos = np.where(gt_commands == ARC_IDX)
            circle_pos = np.where(gt_commands == CIRCLE_IDX)

            args_comp = (gt_args == out_args).astype(np.int)
            all_ext_args_comp.append(args_comp[ext_pos][:, -N_ARGS_EXT:])
            all_line_args_comp.append(args_comp[line_pos][:, :2])
            all_arc_args_comp.append(args_comp[arc_pos][:, :4])
            all_circle_args_comp.append(args_comp[circle_pos][:, [0, 1, 4]])

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

    # define different inference modes
    def reconstruct_vecs(self, cfg):
        cfg.zs = self.encode_vecs(cfg)
        self.decode_zs(cfg)

    def encode_vecs(self, cfg):
        vecs = []
        save_paths = []
        batch_size = cfg.batch_size

        all_zs = {"zs": [], "z_paths": [], "vec_paths": []}

        self.net.eval()
        if cfg.data_root:
            if os.path.isfile(cfg.data_root):
                if cfg.data_root.endswith(".h5") and not cfg.data_root.endswith(
                    "_dec.h5"
                ):
                    with h5py.File(cfg.data_root, "r") as fp:
                        vecs.append(fp["vec"][:])
                    save_paths.append(cfg.data_root)
                    save_name = cfg.data_root.split("/")[-1].split(".")[0]
                else:
                    raise ValueError("Invalid file format")

            elif os.path.isdir(cfg.data_root):
                for file in walk_dir(cfg.data_root):
                    if file.endswith(".h5") and not file.endswith("_dec.h5"):
                        with h5py.File(file, "r") as fp:
                            vecs.append(fp["vec"][:])
                        save_paths.append(file)
                        save_name = os.path.basename(os.path.normpath(cfg.data_root))

            else:
                raise ValueError("Invalid path")

        else:
            raise ValueError("No vecs provided.")

        save_dir = os.path.join(
            self.cfg.exp_dir,
            "results/vecEncodings/",
            self.cfg.ckpt,
            save_name
            + "_"
            + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # encode
        for i in tqdm(range(0, len(vecs), batch_size)):
            with torch.no_grad():
                batch_vec = torch.tensor(
                    np.concatenate(vecs[i : i + batch_size]), dtype=torch.float32
                ).unsqueeze(1)
                batch_vec = batch_vec.cuda()
                batch_z = self.encode(batch_vec, is_batch=True)
                batch_z = batch_z.detach().cpu().numpy()[:, 0, :]

            for j in range(len(batch_vec)):
                zs = batch_z[j]
                file_name = os.path.split(save_paths[i + j])[1]
                save_name = file_name.split(".")[0] + "_zs.h5"
                save_path_zs = os.path.join(save_dir, f"{save_name}")
                try:
                    with h5py.File(save_path_zs, "w") as fp:
                        fp.create_dataset("zs", data=zs, dtype=np.int32)
                    print(f"File: {file_name} encoded.")
                except Exception as e:
                    print(
                        f"File: {file_name} could not be encoded."
                        + str(e.with_traceback)
                    )

                all_zs["zs"].append(zs)
                all_zs["z_paths"].append(save_path_zs)
                all_zs["vec_paths"].append(save_paths[i + j])

        return all_zs

    def decode_zs(self, cfg):
        zs = []
        save_paths = []
        batch_size = cfg.batch_size
        self.net.eval()

        if not hasattr(cfg, "zs"):
            if cfg.z_path:
                if os.path.isfile(cfg.z_path):
                    if cfg.z_path.endswith(".h5") and not cfg.z_path.endswith(
                        "_dec.h5"
                    ):
                        with h5py.File(cfg.z_path, "r") as fp:
                            zs.append(fp["zs"][:])
                        save_paths.append(cfg.z_path)
                    else:
                        raise ValueError("Invalid file format")

                elif os.path.isdir(cfg.z_path):
                    for file in walk_dir(cfg.z_path):
                        if file.endswith(".h5") and not file.endswith("_dec.h5"):
                            with h5py.File(file, "r") as fp:
                                zs.append(fp["zs"][:])
                            save_paths.append(file)

                else:
                    raise ValueError("Invalid path")
            else:
                raise ValueError("No zs provided.")

        else:
            zs = cfg.zs["zs"]
            save_paths = cfg.zs["z_paths"]

        # decode
        for i in tqdm(range(0, len(zs), batch_size)):
            with torch.no_grad():
                batch_z = torch.tensor(
                    np.concatenate(zs[i : i + batch_size]), dtype=torch.float32
                ).unsqueeze(1)
                batch_z = batch_z.cuda()
                outputs = self.decode(batch_z)
                batch_out_vec = self.logits2vec(outputs)

            for j in range(len(batch_z)):
                print("\n*** File: " + save_paths[i + j].split("/")[-1] + " ***\n")
                out_vec = batch_out_vec[j]
                out_vec = trim_vec_EOS(out_vec)

                save_path = save_paths[i + j].split(".")[0]
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
                    for cnt_retries in range(1, cfg.n_checkBrep_retries + 1):
                        print(
                            f"Trying to create a new CAD-Solid. Attempt {cnt_retries}/{cfg.n_checkBrep_retries}"
                        )
                        # print(batch_z.shape, batch_z[j].shape)
                        new_batch_output = self.decode(batch_z[j].unsqueeze(0))
                        out_batch_vec = self.logits2vec(new_batch_output)
                        out_vec = out_batch_vec.squeeze(0)

                        out_vec = trim_vec_EOS(out_vec)

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
                if cfg.expSTEP:
                    try:
                        create_step_file(out_shape, step_save_path)
                    except Exception as e:
                        print(str(e.with_traceback))
                        continue

                if cfg.expPNG or cfg.expGIF:
                    try:
                        png_path = step_save_path.split(".")[0]
                        if step_file_exists(step_save_path):
                            transform(
                                step_save_path,
                                png_path,
                                135,
                                45,
                                "medium",
                                exp_png=cfg.expPNG,
                                make_gif=cfg.expGIF,
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
                            f"Creation of Image-Output for {save_paths[i + j].split('/')[-1]} failed.\n"
                            + str(e.with_traceback)
                        )
                        continue
