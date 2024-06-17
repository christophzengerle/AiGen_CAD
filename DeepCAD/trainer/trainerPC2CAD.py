import datetime
import json
import os
import random
import sys
from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import Parallel, delayed
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from tqdm import tqdm
from utils.step_utils import create_step_file, step_file_exists

from .loss import CADLoss

sys.path.append("..")
from cadlib.macro import *
from cadlib.macro import trim_vec_EOS
from cadlib.visualize import CADsolid2pc, vec2CADsolid
from dataset.vec2pc import convert_vec2pc
from evaluation.pc2cad.evaluate_pc2cad_cd import chamfer_dist
from evaluation.pc2cad.evaluate_pc2cad_gen import (
    compute_cov_mmd,
    jsd_between_point_cloud_sets,
)
from model.pointcloud2cad import PointCloud2CAD
from trainer.scheduler import GradualWarmupScheduler
from utils import read_ply
from utils.file_utils import walk_dir
from utils.step2png import transform

from .base import BaseTrainer
from .trainerAE import TrainerAE
from .trainerPCEncoder import TrainerPCEncoder


class TrainerPC2CAD(BaseTrainer):
    def __init__(self, cfg):
        super(TrainerPC2CAD, self).__init__(cfg)

        self.build_net(self.cfg)

    def build_net(self, cfg):
        self.net = PointCloud2CAD(cfg).cuda()
        if len(cfg.gpu_ids) > 1:
            self.net = nn.DataParallel(self.net)

    def set_loss_function(self):
        self.criterion = CADLoss(self.cfg).cuda()

    def set_optimizer(self, cfg):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            cfg.lr,
            # , betas=(config.beta1, 0.9))
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, cfg.lr_step_size
        )

    # self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, cfg.warmup_step)

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

        if self.cfg.cont:
            print("********* Continue Training from Checkpoint ***********")
        else:
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
                    pbar.set_postfix(
                        OrderedDict({k: v.item() for k, v in loss.items()})
                    )

                self.record_losses(eval_losses, mode="eval")

            if clock.epoch % self.cfg.save_frequency == 0:
                self.test(test_loader)
                self.save_ckpt()

            self.record_and_update_learning_rate()

            clock.tock()

        # if clock.epoch % 10 == 0:
        self.save_ckpt("latest")

    def evaluate(self, data):
        self.net.eval()
        with torch.no_grad():
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
        all_cmd_comp = []
        all_ext_args_comp = []
        all_line_args_comp = []
        all_arc_args_comp = []
        all_circle_args_comp = []

        for i, data in enumerate(pbar):
            with torch.no_grad():
                points = data["points"]
                codes = data["codes"]

                pred = self.forward(points)
                batch_out_vec = self.logits2vec(pred)

                loss = self.criterion(pred, codes)
            test_losses["losses_cmd"].append(loss["loss_cmd"].item())
            test_losses["losses_args"].append(loss["loss_args"].item())

            pbar.set_description(
                "TEST - EPOCH[{}]-[{}] BATCH[{}]-[{}]".format(
                    self.clock.epoch, self.nr_epochs, i, len(test_loader)
                )
            )
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in loss.items()}))

            commands, args = codes["command"], codes["args"]
            gt_commands = commands.squeeze(1).long().detach().cpu().numpy()  # (N, S)
            gt_args = args.squeeze(1).long().detach().cpu().numpy()  # (N, S, n_args)

            out_cmd = batch_out_vec[:, :, 0]
            out_args = batch_out_vec[:, :, 1:]

            mask = (out_cmd == gt_commands)
            cmd_acc = mask.astype(np.int32)
            all_cmd_comp.append(np.mean(cmd_acc))


            corr_gt_cmd = gt_commands[mask]
            ext_pos = np.where(corr_gt_cmd == EXT_IDX)
            line_pos = np.where(corr_gt_cmd == LINE_IDX)
            arc_pos = np.where(corr_gt_cmd == ARC_IDX)
            circle_pos = np.where(corr_gt_cmd == CIRCLE_IDX)


            gt_args = gt_args[mask]
            out_args = out_args[mask]

            args_comp = (np.abs(gt_args == out_args) < ACC_TOLERANCE).astype(np.int32)
            args_comp[ext_pos][:, -N_ARGS_EXT:] = (gt_args[ext_pos][:, -N_ARGS_EXT:] == out_args[ext_pos][:, -N_ARGS_EXT:]).astype(np.int32)
            args_comp[arc_pos][:, :4] = (gt_args[arc_pos][:, :4] == out_args[arc_pos][:, :4]).astype(np.int32)
            
                
            all_ext_args_comp.append(args_comp[ext_pos][:, -N_ARGS_EXT:])
            all_line_args_comp.append(args_comp[line_pos][:, :2])
            all_arc_args_comp.append(args_comp[arc_pos][:, :4])
            all_circle_args_comp.append(args_comp[circle_pos][:, [0, 1, 4]])

        self.record_losses(test_losses, mode="test")

        all_cmd_comp = np.mean(all_cmd_comp)
        all_ext_args_comp = np.concatenate(all_ext_args_comp, axis=0)
        sket_plane_acc = np.mean(all_ext_args_comp[:, :N_ARGS_PLANE])
        sket_trans_acc = np.mean(
            all_ext_args_comp[:, N_ARGS_PLANE : N_ARGS_PLANE + N_ARGS_TRANS]
        )
        extent_one_acc = np.mean(all_ext_args_comp[:, -N_ARGS_EXT_PARAM])
        line_acc = np.mean(np.concatenate(all_line_args_comp, axis=0))
        arc_acc = np.mean(np.concatenate(all_arc_args_comp, axis=0))
        circle_acc = np.mean(np.concatenate(all_circle_args_comp, axis=0))

        self.test_tb.add_scalar(
            "cmd_acc",
            all_cmd_comp,
            global_step=self.clock.epoch,
        )

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
        
        # print("args_acc" + str(all_cmd_comp) + "line" + str(line_acc) +  "arc" + str(arc_acc) +  "circle" + str(circle_acc) +  "plane" + str(sket_plane_acc) +  "trans" + str(sket_trans_acc) +  "extent" + str(extent_one_acc) )

    def eval_model_acc(self, test_loader):
        print("********** Calculating Accuracy-Metrics **********")
        self.net.eval()
        save_dir = os.path.join(
            self.cfg.exp_dir,
            "evaluation/accuracy/",
            self.cfg.ckpt,
            str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # overall accuracy
        avg_cmd_acc = []  # ACC_cmd
        avg_param_acc = []  # ACC_param

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
                batch_cmd_acc = []  # ACC_cmd
                batch_param_acc = []  # ACC_param

                points = data["points"]
                codes = data["codes"]

                gt_cmd = codes["command"].detach().cpu().numpy()
                gt_param = codes["args"].detach().cpu().numpy()

                pred = self.forward(points)
                batch_out_vec = self.logits2vec(pred)

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

                    if (
                        out_cmd[i][j] == gt_cmd[i][j]
                    ):  # NOTE: only account param acc for correct cmd
                        tole_acc = (
                            np.abs(out_param[i][j] - gt_param[i][j]) < ACC_TOLERANCE
                        ).astype(np.int32)
                        # filter param that do not need tolerance (i.e. requires strictly equal)
                        if cmd == EXT_IDX:
                            tole_acc[-2:] = (out_param[i][j] == gt_param[i][j]).astype(
                                np.int32
                            )[-2:]
                        elif cmd == ARC_IDX:
                            tole_acc[3] = (out_param[i][j] == gt_param[i][j]).astype(
                                np.int32
                            )[3]

                        valid_param_acc = tole_acc[
                            args_mask[cmd].astype(np.bool_)
                        ].tolist()
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
                "TEST ACCURACY - BATCH[{}]-[{}]".format(batch_nr, len(test_loader))
            )
            pbar.set_postfix(
                OrderedDict(
                    {
                        "CMD-Acc": np.mean(batch_cmd_acc),
                        "Args-Acc": np.mean(batch_param_acc),
                    }
                )
            )

        save_path = os.path.join(save_dir, "test_acc_stats.txt")
        fp = open(save_path, "w")
        print("Tolerance: ", ACC_TOLERANCE, file=fp)
        # overall accuracy (averaged over all data)
        avg_cmd_acc = np.mean(avg_cmd_acc)
        print("avg command acc (ACC_cmd):", avg_cmd_acc, file=fp)

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
            print(
                ALL_COMMANDS[i] + " param acc:",
                each_param_acc[i][args_mask[i].astype(np.bool_)],
                file=fp,
            )
        fp.close()

        with open(save_path, "r") as fp:
            res = fp.readlines()
            for l in res:
                print(l, end="")

    def eval_model_chamfer_dist(self, test_loader):
        print("********** Calculating Chamfer-Distance **********")
        self.net.eval()
        # PROCESS_PARALLEL = True if self.cfg.num_workers > 1 else False
        PROCESS_PARALLEL = False

        save_dir = os.path.join(
            self.cfg.exp_dir,
            "evaluation/chamfer_dist/",
            self.cfg.ckpt,
            str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, "test_cd_stats.txt")

        def process_one_cd(out_vec, gt_pc, data_id):
            out_pc = convert_vec2pc(out_vec, data_id, self.cfg.n_points)
            if out_pc is None:
                return None

            # sample_idx = random.sample(list(range(gt_pc.shape[0])), self.cfg.n_points if self.cfg.n_points < pc.shape[0] else pc.shape[0])
            # gt_pc = gt_pc[sample_idx]
            cd = chamfer_dist(gt_pc, out_pc)
            return cd

        n_dists = len(test_loader.dataset)
        valid_dists = []
        pbar = tqdm(test_loader)
        for batch_nr, data in enumerate(pbar):
            with torch.no_grad():
                print("processing Batch - [{}]-[{}]".format(batch_nr, len(test_loader)))
                points = data["points"]  # Input Point Cloud

                pred = self.forward(points)
                batch_out_vec = self.logits2vec(pred)
                points = points.detach().cpu().numpy()

            if PROCESS_PARALLEL:
                res = Parallel(n_jobs=self.cfg.num_workers, verbose=2)(
                    delayed(process_one_cd)(trim_vec_EOS(out_vec), gt_pc, data_id)
                    for gt_pc, out_vec, data_id in zip(
                        points,
                        batch_out_vec,
                        data["ids"],
                    )
                )
                valid_dists.extend([x for x in res if x is not None])
            else:
                for i in range(len(points)):
                    gt_pc = points[i]
                    out_vec = batch_out_vec[i]
                    out_vec = trim_vec_EOS(out_vec)
                    data_id = data["ids"][i]

                    res = process_one_cd(out_vec, gt_pc, data_id)
                    if res is not None:
                        valid_dists.append(res)

        valid_dists = sorted(valid_dists)
        print("top 20 largest error:")
        print(valid_dists[-20:][::-1])
        n_valid = len(valid_dists)
        n_invalid = n_dists - n_valid

        avg_dist = np.mean(valid_dists)
        trim_avg_dist = np.mean(valid_dists[int(n_valid * 0.1) : -int(n_valid * 0.1)])
        med_dist = np.median(valid_dists)

        print("#####" * 10)
        print(
            "total:",
            n_dists,
            "\t invalid:",
            n_invalid,
            "\t invalid ratio:",
            n_invalid / n_dists,
        )
        print(
            "avg dist:",
            avg_dist,
            "trim_avg_dist:",
            trim_avg_dist,
            "med dist:",
            med_dist,
        )
        with open(save_path, "wr") as fp:
            print("#####" * 10, file=fp)
            print(
                "total:",
                n_dists,
                "\t invalid:",
                n_invalid,
                "\t invalid ratio:",
                n_invalid / n_dists,
                file=fp,
            )
            print(
                "avg dist:",
                avg_dist,
                "trim_avg_dist:",
                trim_avg_dist,
                "med dist:",
                med_dist,
                file=fp,
            )

    def eval_model_cov_mmd_jsd(self, test_loader):
        print("********** Calculating COV - MMD - JSD **********")
        self.net.eval()
        save_dir = os.path.join(
            self.cfg.exp_dir,
            "evaluation/chamfer_dist/",
            self.cfg.ckpt,
            str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, "test_gen_stats.txt")

        n_measures = len(test_loader.dataset)
        result_list = []
        pbar = tqdm(test_loader)
        for batch_nr, data in enumerate(pbar):
            with torch.no_grad():
                print("processing Batch [{}]-[{}]".format(batch_nr, len(test_loader)))
                points = data["points"]  # Input Point Cloud

                pred = self.forward(points)
                batch_out_vec = self.logits2vec(pred)

            gt_pcs = points.detach().cpu().numpy()

            proc_pcs = []
            for i in range(len(points)):
                out_vec = batch_out_vec[i]
                out_vec = trim_vec_EOS(out_vec)
                data_id = data["ids"][i]

                out_pc = convert_vec2pc(out_vec, data_id, self.cfg.n_points)
                proc_pcs.append(out_pc)

            gen_pcs = [pc for pc in proc_pcs if pc is not None]
            gen_pcs = np.stack(gen_pcs, axis=0)

            jsd = jsd_between_point_cloud_sets(gen_pcs, gt_pcs, in_unit_sphere=False)

            gen_pcs = torch.tensor(gen_pcs).cuda()
            ref_pcs = torch.tensor(ref_pcs).cuda()
            result = compute_cov_mmd(gen_pcs, ref_pcs, batch_size=len(gt_pcs))
            result.update({"JSD": jsd})

            result_list.append(result)

        n_valid = len(result_list)
        n_invalid = n_measures - n_valid
        avg_result = {}
        for k in result_list[0].keys():
            avg_result.update({"avg-" + k: np.mean([x[k] for x in result_list])})
        print("#####" * 10)
        print(
            "total:",
            n_measures,
            "\t invalid:",
            n_invalid,
            "\t invalid ratio:",
            n_invalid / n_measures,
        )
        print("average result:")
        print(avg_result)
        with open(save_path, "wr") as fp:
            print("#####" * 10, file=fp)
            print(
                "total:",
                n_measures,
                "\t invalid:",
                n_invalid,
                "\t invalid ratio:",
                n_invalid / n_measures,
                file=fp,
            )
            print("average result:", file=fp)
            print(avg_result, file=fp)

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



        if self.cfg.output is not None:
            if os.path.isfile(self.cfg.output):
                out_dir = os.dirname(self.cfg.output)
            elif os.path.isdir(self.cfg.output):
                out_dir = self.cfg.output
            else:
                try:
                    os.makedirs(self.cfg.output)
                    out_dir = self.cfg.output
                except Exception as e:
                    print("Output-path is invalid. Using default path.")
                    
        out_dir = os.path.join(
            self.cfg.output,
            self.cfg.ckpt,
            save_name
            + "_"
            + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
        )

        save_dir = out_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print(f"File-List: {file_list}")
        if len(file_list) > 0:
            valid_preds = 0
            pbar = tqdm(file_list)
            for i, pc_path in enumerate(pbar):
                with torch.no_grad():
                    # retrieve Data
                    pc = read_ply(pc_path)
                    sample_idx = random.sample(
                        list(range(pc.shape[0])),
                        self.cfg.n_points if self.cfg.n_points < pc.shape[0] else pc.shape[0],
                    )
                    pc = pc[sample_idx]
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

                    pred = self.forward(pc)
                    batch_out_vec = self.logits2vec(pred)
                    out_vec = batch_out_vec.squeeze(0)
                    out_vec = trim_vec_EOS(out_vec)

                print(out_vec)
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

                    # check generated CAD-Shape and save it
                    out_shape = None
                    is_valid_BRep = False
                    try:
                        out_shape = vec2CADsolid(out_vec)
                        is_valid_BRep = True
                        valid_preds += 1
                    except Exception as e:
                        print(str(e))

                        # check generated CAD-Shape
                        # if invalid -> generate again
                        for cnt_retries in range(1, self.cfg.n_checkBrep_retries + 1):
                            print(
                                f"Trying to create a new CAD-Solid. Attempt {cnt_retries}/{self.cfg.n_checkBrep_retries}"
                            )

                            new_batch_output = self.forward(pc)
                            out_batch_vec = self.logits2vec(new_batch_output)
                            out_vec = out_batch_vec.squeeze(0)
                            out_vec = trim_vec_EOS(out_vec)

                            try:
                                out_shape = vec2CADsolid(out_vec)
                                analyzer = BRepCheck_Analyzer(out_shape)
                                if analyzer.IsValid():
                                    print("Valid BRep-Model detected.")
                                    is_valid_BRep = True
                                    valid_preds += 1
                                    break
                                else:
                                    print("invalid BRep-Model detected.")
                                    continue

                            except Exception as e:
                                print(str(e))
                                continue

                    if not is_valid_BRep:
                        print("Could not create valid BRep-Model!")
                        continue

                    save_path_vec = save_path + "_dec.h5"
                    with h5py.File(save_path_vec, "w") as fp:
                        fp.create_dataset("out_vec", data=out_vec, dtype=np.int32)

                    step_save_path = save_path + "_dec.step"
                    print(step_save_path)
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
        print(f"********** {valid_preds} / {len(file_list)} succeeded ***********")
        
        return step_save_path

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

    def load_ckpt(self):
        """load checkpoint from saved checkpoint"""
        if self.cfg.load_modular_ckpt:
            pcEnc_model_dir = os.path.join(
                self.cfg.proj_dir, "pce", self.cfg.pce_exp_name, "model"
            )
            ae_model_dir = os.path.join(
                self.cfg.proj_dir, "ae", self.cfg.ae_exp_name, "model"
            )

            pcEnc_load_path = os.path.join(
                pcEnc_model_dir, "{}.pth".format(self.cfg.pce_ckpt)
            )
            ae_load_path = os.path.join(ae_model_dir, "{}.pth".format(self.cfg.ae_ckpt))

            if not os.path.exists(pcEnc_load_path):
                raise ValueError(
                    "PcEncoder-Checkpoint {} not exists.".format(pcEnc_load_path)
                )
            if not os.path.exists(ae_load_path):
                raise ValueError(
                    "AutoEncoder-Checkpoint {} not exists.".format(ae_load_path)
                )

            ae_checkpoint = torch.load(ae_load_path)
            print("Loading AE-checkpoint from {} ...".format(ae_load_path))
            if isinstance(self.net, nn.DataParallel):
                self.net.module.load_state_dict(
                    ae_checkpoint["model_state_dict"], strict=False
                )
            else:
                self.net.load_state_dict(
                    ae_checkpoint["model_state_dict"], strict=False
                )

            pcEnc_checkpoint = torch.load(pcEnc_load_path)
            print("Loading pcEnc-checkpoint from {} ...".format(pcEnc_load_path))
            if isinstance(self.net, nn.DataParallel):
                self.net.pc_enc.module.load_state_dict(
                    pcEnc_checkpoint["model_state_dict"]
                )
            else:
                self.net.pc_enc.load_state_dict(pcEnc_checkpoint["model_state_dict"])

            print(
                "Checkpoints for pretrained AutoEncoder & PointCloud-Encoder loaded successfully."
            )

        else:
            load_path = os.path.join(self.model_dir, "{}.pth".format(self.cfg.ckpt))
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

            print("Checkpoint loaded successfully.")

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
