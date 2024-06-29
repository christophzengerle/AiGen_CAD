import argparse
import json
import os
import shutil

from utils import ensure_dirs

class ConfigPCEncoder(object):
    def __init__(self):
        _, args = self.parse()

        self.z_path = args.z_path
        self.pc_root = args.pc_root
        self.split_path = args.split_path
        self.exp_dir = os.path.join(args.proj_dir, "pce", args.exp_name)
        self.log_dir = os.path.join(self.exp_dir, "log")
        self.model_dir = os.path.join(self.exp_dir, "model")
        self.gpu_ids = args.gpu_ids

        # set as attributes
        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.set_configuration()

        print("----Pc-Experiment Configuration-----")
        for k, v in self.__dict__.items():
            print("{0:20}".format(k), v)

        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        self.exec = args.exec
        self.mode = args.mode
        self.cont = args.cont
        self.ckpt = args.ckpt
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
            # response = input("Experiment log/model already exists, overwrite? (y/n) ")
            response = "y"
            if response != "y":
                exit()
            shutil.rmtree(self.exp_dir)
        ensure_dirs([self.log_dir, self.model_dir])
        if args.exec == "train":
            with open("{}/config.txt".format(self.exp_dir), "w") as f:
                json.dump(self.__dict__, f, indent=2)

    def set_configuration(self):

        self.lr = 1e-3  # initial LR
        self.warmup_step = 100  # Nr warmup Epochs, LR will increase from 0 to self.lr
        # self.lr_step_size = 100  # Nr Epochs after wich LR will be decresed
        # self.beta1 = 0.5
        self.grad_clip = None
        self.noiseAmount = 0.02

        self.val_frequency = 1
        self.save_frequency = 50

        self.expSourcePNG = True
        
        self.faulty_cad_models_path = "dataset/faulty_cad_models.json"

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--exec",
            "-e",
            type=str,
            choices=["train", "eval", "inf"],
            default="test",
            help="different execution modes for Pc-Encoder: train - Trains on Train and Eval dataset, eval - Evaluate on Test dataset, Inf - Inference on own data",
        )
        parser.add_argument(
            "--mode",
            "-m",
            type=str,
            choices=["enc", "rec"],
            default="rec",
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
            "--split_path",
            type=str,
            default="data/train_val_test_split.json",
            help="path to train-val-test split",
        )
        parser.add_argument(
            "--z_path",
            type=str,
            default="data/cad_all_zs.h5",
            help="path to zs.h5 file containing ground truth shape codes",
        )
        parser.add_argument(
            "--output",
            "-o",
            dest="output",
            type=str,
            default=None,
            help="specify output-path for results in inference mode",
        )
        parser.add_argument(
            "--exp_name",
            type=str,
            required=False,
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
        parser.add_argument(
            "--warmup_step",
            type=int,
            default=10,
            help="step size for learning rate warm up",
        )
        parser.add_argument("--batch_size", type=int, default=128, help="batch size")
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
