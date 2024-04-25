import argparse
import json
import os
import shutil

from utils import ensure_dirs


class ConfigPcEncoder(object):
    def __init__(self):
        self.set_configuration()
        _, args = self.parse()

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        self.data_root = os.path.join("../data", "all_zs.h5")
        self.pc_root = args.pc_root
        self.split_path = args.split_path
        self.exp_dir = os.path.join(args.proj_dir, args.exp_name, "pc2cad")
        self.log_dir = os.path.join(self.exp_dir, "log")
        self.model_dir = os.path.join(self.exp_dir, "model")
        self.gpu_ids = args.gpu_ids

        if args.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

        self.test = args.test
        self.cont = args.cont
        self.ckpt = args.ckpt
        self.n_samples = args.n_samples

        if (not args.test) and args.cont is not True and os.path.exists(self.exp_dir):
            response = input("Experiment log/model already exists, overwrite? (y/n) ")
            if response != "y":
                exit()
            shutil.rmtree(self.exp_dir)
        ensure_dirs([self.log_dir, self.model_dir])
        if not args.test:
            os.system("cp pc2cad.py {}".format(self.exp_dir))
            with open("{}/config.txt".format(self.exp_dir), "w") as f:
                json.dump(self.__dict__, f, indent=2)

    def set_configuration(self):
        self.n_points = 4096
        self.batch_size = 256
        self.num_workers = 8
        self.nr_epochs = 1000
        self.lr = 1e-4
        self.lr_step_size = 50
        # self.beta1 = 0.5
        self.grad_clip = None
        self.noise = 0.02

        self.save_frequency = 100
        self.val_frequency = 10

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--proj_dir",
            type=str,
            default="proj_log",
            help="path to project folder where models and logs will be saved",
        )
        parser.add_argument(
            "--pc_root",
            type=str,
            default="path_to_pc_data",
            help="path to point clouds data folder",
        )
        parser.add_argument(
            "--split_path",
            type=str,
            default="data/train_val_test_split.json",
            help="path to train-val-test split",
        )
        parser.add_argument(
            "--exp_name", type=str, required=True, help="name of this experiment"
        )
        parser.add_argument("--ae_ckpt", type=str, help="desired checkpoint to restore")
        parser.add_argument(
            "--continue",
            dest="cont",
            action="store_true",
            help="continue training from checkpoint",
        )
        parser.add_argument(
            "--ckpt",
            type=str,
            default="latest",
            required=False,
            help="desired checkpoint to restore",
        )
        parser.add_argument("--test", action="store_true", help="test mode")
        parser.add_argument(
            "--n_samples",
            type=int,
            default=100,
            help="number of samples to generate when testing",
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
