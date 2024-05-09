import os
import glob
from dataset import vec2pc
import numpy as np
import h5py
from joblib import Parallel, delayed
import argparse
import sys
sys.path.append("..")
from utils import write_ply
from cadlib.visualize import vec2CADsolid, CADsolid2pc

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default=None, required=True)
    parser.add_argument('--n_points', type=int, default=2000)
    args = parser.parse_args()
    return parser, args

def process_one(path, n_points):
    data_id = path.split("/")[-1]

    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    if os.path.exists(save_path):
        return

    # print("[processing] {}".format(data_id))
    with h5py.File(path, 'r') as fp:
        out_vec = fp["out_vec"][:].astype(np.float)

    out_pc = vec2pc(out_vec, data_id, n_points)
    
    save_path = os.path.join(SAVE_DIR, data_id + ".ply")
    write_ply(out_pc, save_path)

def main():
    parser, args = parse()
    global SAVE_DIR
    SAVE_DIR = args.src + '_pc'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    all_paths = glob.glob(os.path.join(args.src, "*.h5"))
    Parallel(n_jobs=8, verbose=2)(delayed(process_one)(x, args.n_points) for x in all_paths)
    
if __name__ == '__main__':
    main()
