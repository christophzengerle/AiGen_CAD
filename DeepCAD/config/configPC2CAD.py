import argparse
import json
import os
import shutil

from cadlib.macro import *
from utils import ensure_dirs


class ConfigPC2CAD(object):
    def __init__(self):
        self.set_configuration()
        _, args = self.parse()

        # set as attributes
        print("----Pc-Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        self.pc_root = args.pc_root
        self.split_path = args.split_path
        self.exp_dir = os.path.join(args.proj_dir, "pc2cad", args.exp_name)
        self.log_dir = os.path.join(self.exp_dir, "log")
        self.model_dir = os.path.join(self.exp_dir, "model")
        self.gpu_ids = args.gpu_ids

        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        self.exec = args.exec
        self.cont = args.cont
        self.ckpt = args.ckpt
        self.pce_ckpt = args.pce_ckpt
        self.ae_ckpt = args.ae_ckpt

        self.num_workers = args.num_workers
        self.n_points = args.n_points
        self.batch_size = args.batch_size
        self.nr_epochs = args.nr_epochs

        if (
            (args.exec == "train")
            and args.cont is not True
            and os.path.exists(self.exp_dir)
        ):
            response = input("Experiment log/model already exists, overwrite? (y/n) ")
            if response != "y":
                exit()
            shutil.rmtree(self.exp_dir)
        ensure_dirs([self.log_dir, self.model_dir])
        if args.exec == "train":
            with open("{}/config.txt".format(self.exp_dir), "w") as f:
                json.dump(self.__dict__, f, indent=2)

    def set_configuration(self):

        # Pc-Encoder
        self.lr = 1e-3  # initial LR
        self.lr_step_size = 25  # Nr Epochs after wich LR will be decresed
        # self.beta1 = 0.5
        # self.grad_clip = None
        self.noiseAmount = 0.025

        # Autoencoder
        self.args_dim = ARGS_DIM  # 256
        self.n_args = N_ARGS
        self.n_commands = len(ALL_COMMANDS)  # line, arc, circle, EOS, SOS

        self.n_layers = 4  # Number of Encoder blocks
        self.n_layers_decode = 4  # Number of Decoder blocks
        self.n_heads = 8  # Transformer config: number of heads
        self.dim_feedforward = 512  # Transformer config: FF dimensionality
        self.d_model = 256  # Transformer config: model dimensionality
        self.dropout = 0.1  # Dropout rate used in basic layers and Transformers
        self.dim_z = 256  # Latent vector dimensionality
        self.use_group_emb = True

        self.max_n_ext = MAX_N_EXT
        self.max_n_loops = MAX_N_LOOPS
        self.max_n_curves = MAX_N_CURVES

        self.max_num_groups = 30
        self.max_total_len = MAX_TOTAL_LEN

        self.n_checkBrep_retries = 5  # num retries to create valid CAD model

        self.loss_weights = {"loss_cmd_weight": 1.0, "loss_args_weight": 2.0}

        # General Settings
        self.save_frequency = 10
        self.val_frequency = 5

        self.expSourcePNG = True

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--exec",
            "-e",
            type=str,
            choices=["train", "test", "inf"],
            default="test",
            help="different execution modes for Pc-Encoder: train - Trains on Train and Eval dataset, test - Test on Test dataset, Inf - Inference on own data",
        )
        parser.add_argument(
            "--mode",
            "-m",
            type=str,
            choices=["test", "acc"],
            default="acc",
            help="choose different execution modes: enc - encode point clouds to latent vecs, rec - reconstruct CAD models out of point clouds",
        )
        parser.add_argument(
            "--proj_dir",
            type=str,
            default="proj_log",
            help="path to project folder where models and logs will be saved",
        )
        parser.add_argument(
            "--pc_root",
            type=str,
            default="data/cad_pc",
            help="file- or folder-path to point cloud data",
        )
        parser.add_argument(
            "--data_root", type=str, default="data", help="path to source data folder"
        )
        parser.add_argument(
            "--split_path",
            type=str,
            default="data/train_val_test_split.json",
            help="path to train-val-test split",
        )
        parser.add_argument(
            "--exp_name",
            type=str,
            required=True,
            default="pcEncoder",
            help="name of this experiment",
        )
        parser.add_argument(
            "--ckpt",
            type=str,
            default="latest",
            required=False,
            help="desired Pc-Encoder checkpoint to restore",
        )
        parser.add_argument(
            "--pce_exp_name",
            type=str,
            required=False,
            default="pretrained",
            help="name of PointCloud experiment",
        )
        parser.add_argument(
            "--pce_ckpt",
            type=str,
            default="latest",
            required=False,
            help="desired Pointcloud checkpoint to restore",
        )
        parser.add_argument(
            "--ae_exp_name",
            type=str,
            required=False,
            default="pretrained",
            help="name of Autoencoder experiment",
        )
        parser.add_argument(
            "--ae_ckpt",
            type=str,
            default="latest",
            required=False,
            help="desired Autoencoder checkpoint to restore",
        )
        parser.add_argument(
            "--continue",
            "-cont",
            dest="cont",
            action="store_true",
            help="continue training from checkpoint",
        )
        parser.add_argument(
            "--n_points",
            type=int,
            default=4096,
            help="number of points to sample from point cloud",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=8,
            help="number of workers for data loading",
        )
        parser.add_argument(
            "--nr_epochs",
            type=int,
            default=100,
            help="total number of epochs to train",
        )
        parser.add_argument("--batch_size", type=int, default=128, help="batch size")
        parser.add_argument(
            "--lr", type=float, default=1e-3, help="initial learning rate"
        )
        parser.add_argument(
            "--grad_clip", type=float, default=1.0, help="initial learning rate"
        )
        parser.add_argument(
            "--warmup_step",
            type=int,
            default=2000,
            help="step size for learning rate warm up",
        )
        parser.add_argument(
            "--noise",
            action="store_true",
            default=False,
            help="train model with noisy data",
        )
        parser.add_argument(
            "--expSTEP",
            action="store_true",
            default=False,
            help="export step file for decoded CAD model",
        )
        parser.add_argument(
            "--expPNG",
            action="store_true",
            default=False,
            help="export png file for decoded CAD model",
        )
        parser.add_argument(
            "--expGIF",
            action="store_true",
            default=False,
            help="export gif file for decoded CAD model",
        )
        parser.add_argument(
            "-g",
            "--gpu_ids",
            type=str,
            default="0",
            help="gpu to use, e.g. 0  0,1,2. CPU not supported.",
        )
        args = parser.parse_args()
        return parser, args
