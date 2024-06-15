import argparse
import multiprocessing
import os
import numpy as np
import sys
import trimesh

sys.path.append("../../../src")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="source folder", required=True)
    parser.add_argument("--dest", type=str, default="png_files", help="destination folder")
    args = parser.parse_args()
    return args


def setup_dir(source_folder, destination_folder):
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Path {source_folder} does not exist")

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)


def walk_dir(dir):
    file_list = []
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            file_list.append(os.path.normpath(os.path.join(subdir, file)))
    return file_list


def transform(file_path, outfile):
    # print('filepath', file_path)
    if file_path.endswith(".ply"):
        m = trimesh.load_mesh(file_path)
    elif file_path.endswith(".step"):
        m = trimesh.Trimesh(
            **trimesh.interfaces.gmsh.load_gmsh(
                file_name=file_path,
                gmsh_args=[
                    ("Mesh.Algorithm", 1),  # Different algorithm types, check them out
                    (
                        "Mesh.CharacteristicLengthFromCurvature",
                        50,
                    ),  # Tuning the smoothness, + smoothness = + time
                    ("General.NumThreads", 10),  # Multithreading capability
                    ("Mesh.MinimumCirclePoints", 32),
                ],
            )
        )

    outfile = outfile + ".obj"
    m.export(outfile, file_type='obj')
    
    return outfile

def main():
    args = parse()
    setup_dir(args.src, args.dest)

    if os.path.isfile(args.src):
        if args.src.endswith(".step"):
            objfiles = [args.src]

    elif os.path.isdir(args.src):
        objfiles = [
            file
            for file in walk_dir(args.src)
            if file.endswith(".step")
        ]

    else:
        raise ValueError("No valid source file type.")

    for i, file_path in enumerate(objfiles):
        path, file = os.path.split(file_path)
        outfile = os.path.join(args.dest, file).split('.')[0]

        p = multiprocessing.Process(
            target=transform, args=(file_path, outfile)
        )
        p.start()
        p.join(60)

        if p.is_alive():
            print("still running")
            p.terminate()
            p.join()

        print(f"Progress: {(i + 1) / len(objfiles) * 100}")


if __name__ == '__main__':
    main()
