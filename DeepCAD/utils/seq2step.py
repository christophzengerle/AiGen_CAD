import argparse
import json
import os
import sys

import h5py
import numpy as np
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.DataExchange import write_step_file
from step_utils import create_step_file

sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.visualize import create_CAD, vec2CADsolid
from utils.file_utils import walk_dir


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="source folder", required=True)
    parser.add_argument(
        "--dest", type=str, default="step_files", help="destination folder"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="h5",
        choices=["h5", "json"],
        help="Source file format",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="use opencascade analyzer to filter invalid model",
    )
    args = parser.parse_args()
    return args


def setup_dir(source_folder, destination_folder):
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Path {source_folder} does not exist")

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)


def main():
    failed_files = []
    args = parse()
    setup_dir(args.src, args.dest)

    if os.path.isfile(args.src):
        if args.src.endswith("." + args.type):
            seq_files = [args.src]

    elif os.path.isdir(args.src):
        seq_files = [
            file for file in walk_dir(args.src) if file.endswith("." + args.type)
        ]

    else:
        raise ValueError("No valid source file type.")

    for file_path in seq_files:
        path, file = os.path.split(file_path)
        outfile = os.path.join(args.dest, file).replace(file.split(".")[-1], "step")

        try:
            if args.type == "h5":
                with h5py.File(file_path, "r") as fp:
                    out_vec = fp["out_vec"][:].astype(np.float64)
                    out_shape = vec2CADsolid(out_vec)
            else:
                with open(file_path, "r") as fp:
                    data = json.load(fp)
                cad_seq = CADSequence.from_dict(data)
                cad_seq.normalize()
                out_shape = create_CAD(cad_seq)

            if args.check:
                analyzer = BRepCheck_Analyzer(out_shape)
                if not analyzer.IsValid():
                    raise ValueError("Invalid model.")

            create_step_file(out_shape, outfile)

        except Exception as e:
            print(f"{file_path} load and create failed.", str(e))
            failed_files.append(file_path)
            continue

    with open(os.path.join(args.dest, "failed_files.txt"), "w") as f:
        f.write("\n".join(failed_files))


if __name__ == "__main__":
    main()
