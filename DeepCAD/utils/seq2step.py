import argparse
import json
import os
import sys

import h5py
import numpy as np
from OCC.Extend.DataExchange import write_step_file

sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.visualize import create_CAD, vec2CADsolid

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, help="source folder", required=True)
parser.add_argument("--fdepth", type=int, default=1, help="source folder depth")
parser.add_argument("--dest", type=str, default="step_files", help="destination folder")
parser.add_argument(
    "--type",
    type=str,
    default="h5",
    choices=["h5", "json"],
    help="Source file format",
)
args = parser.parse_args()


def setup_dir(source_folder, destination_folder):
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Path {source_folder} does not exist")

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)


failed_files = []
setup_dir(args.src, args.dest)

src_dir = args.src
if args.fdepth == 1:
    if args.type == "h5":
        seq_files = [
            os.path.join(args.src, file)
            for file in os.listdir(args.src)
            if file.endswith(".h5")
        ]
    elif args.type == "json":
        seq_files = [
            os.path.join(args.src, file)
            for file in os.listdir(args.src)
            if file.endswith(".json")
        ]
elif args.fdepth == 2:
    if args.type == "h5":
        seq_files = [
            os.path.join(args.src, folder, file)
            for folder in list(os.walk(args.src))[0][1]
            for file in os.listdir(os.path.join(args.src, folder))
            if file.endswith(".h5")
        ]
    elif args.type == "json":
        seq_files = [
            os.path.join(args.src, folder, file)
            for folder in list(os.walk(args.src))[0][1]
            for file in os.listdir(os.path.join(args.src, folder))
            if file.endswith(".json")
        ]
else:
    raise Exception(f"fdepth of {args.fdepth} not implemented yet")


for file_path in seq_files:
    path, file = os.path.split(file_path)
    outfile = os.path.join(args.dest, file).replace(file.split(".")[-1], "step")

    if args.fdepth == 2:
        folder = os.path.split(path)[1]
        if not os.path.exists(os.path.join(args.dest, folder)):
            os.mkdir(os.path.join(args.dest, folder))
        outfile = os.path.join(args.dest, folder, file).replace(
            file.split(".")[-1], "step"
        )

    try:
        if args.type == "h5":
            with h5py.File(path, "r") as fp:
                out_vec = fp["vec"][:].astype(np.float64)
                out_shape = vec2CADsolid(out_vec)
        else:
            with open(path, "r") as fp:
                data = json.load(fp)
            cad_seq = CADSequence.from_dict(data)
            cad_seq.normalize()
            out_shape = create_CAD(cad_seq)

    except Exception as e:
        print(f"{file_path} load and create failed.")
        failed_files.append(file_path)
        continue

    write_step_file(out_shape, outfile)

with open(os.path.join(args.dest, "failed_files.txt"), "w") as f:
    f.write("\n".join(failed_files))
