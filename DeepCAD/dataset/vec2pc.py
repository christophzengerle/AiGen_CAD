import argparse
import glob
import json
import os
import random
import sys

import h5py
import numpy as np
from evaluation.pc2cad.evaluate_pc2cad_cd import normalize_pc
from joblib import Parallel, delayed
from trimesh.sample import sample_surface

sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.visualize import CADsolid2pc, create_CAD, vec2CADsolid
from utils.pc_utils import read_ply, write_ply



DATA_ROOT = "../data"
RAW_DATA = os.path.join(DATA_ROOT, "cad_vec")
RECORD_FILE = os.path.join(DATA_ROOT, "train_val_test_split.json")

N_POINTS = 8096  # 4096
WRITE_NORMAL = False
SAVE_DIR = os.path.join(DATA_ROOT, "cad_pc")

INVALID_IDS = []


def convert_vec2pc(vec, data_id, n_points):
    shape = vec2CADsolid(vec)


    out_pc = CADsolid2pc(shape, n_points, data_id)


    if np.max(np.abs(out_pc)) > 2:  # normalize out-of-bound data
        out_pc = normalize_pc(out_pc)

    return out_pc


def process_one(data_id):
    if data_id in INVALID_IDS:
        print("skip {}: in invalid id list".format(data_id))
        return

    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    # if os.path.exists(save_path):
    #     print("skip {}: file already exists".format(data_id))
    #     return

    # print("[processing] {}".format(data_id))
    json_path = os.path.join(RAW_DATA, data_id + ".json")
    with open(json_path, "r") as fp:
        data = json.load(fp)

    try:
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        shape = create_CAD(cad_seq)
    except Exception as e:
        print("create_CAD failed:", data_id)
        return None

    try:
        out_pc = CADsolid2pc(shape, N_POINTS, data_id.split("/")[-1])
    except Exception as e:
        print("convert point cloud failed:", data_id)
        return None

    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    truck_dir = os.path.dirname(save_path)
    if not os.path.exists(truck_dir):
        os.makedirs(truck_dir)

    write_ply(out_pc, save_path)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only_test", action="store_true", help="only convert test data")
    args = parser.parse_args()
    return parser, args

def main():
    parser, args = parse()
    with open(RECORD_FILE, "r") as fp:
        all_data = json.load(fp)
        
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # process_one(all_data["train"][3])
    # exit()

    if not args.only_test:
        Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["train"])
        Parallel(n_jobs=10, verbose=2)(
            delayed(process_one)(x) for x in all_data["validation"]
        )
    Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["test"])

if __name__ == '__main__':
    main()