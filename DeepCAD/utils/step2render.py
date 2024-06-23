import argparse
import math
import multiprocessing
import os
import sys
from io import BytesIO

import imageio
import numpy as np
import trimesh
from file_utils import walk_dir
from obj_utils import create_mesh_from_step
from PIL import Image
from pyvirtualdisplay import Display
from trimesh import transformations

res = {"high": 1200, "medium": 600, "low": 300}


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="source folder", required=True)
    parser.add_argument(
        "--dest", type=str, default="png_files", help="destination folder"
    )
    parser.add_argument("--ele", type=int, default=45, help="camera elevation")
    parser.add_argument("--rot", type=int, default=-45, help="camera rotation")
    parser.add_argument("--png", default=False, action="store_true", help="make png")
    parser.add_argument("--gif", default=False, action="store_true", help="make gif")
    parser.add_argument("--obj", default=False, action="store_true", help="make obj")
    parser.add_argument(
        "--qual",
        type=str,
        default="low",
        help="camera rotation",
        choices=["low", "medium", "high"],
    )
    args = parser.parse_args()
    return args


def setup_dir(source_folder, destination_folder):
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Path {source_folder} does not exist")

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)


def transform(
    file_path,
    outfile,
    rotation,
    elevation,
    quality,
    exp_png=False,
    make_gif=False,
    exp_obj=False,
):
    # print('filepath', file_path)

    # init virtual display
    display = Display(visible=0)
    display.start()

    if file_path.endswith(".ply") or file_path.endswith(".obj"):
        mesh = trimesh.load_mesh(file_path)
    elif file_path.endswith(".step"):
        mesh = create_mesh_from_step(file_path)

    else:
        raise ValueError("Invalid File-Type {}.".format(file_path.split(".")[-1]))

    if exp_png:
        export_png(mesh, outfile, rotation, elevation, quality)


    if make_gif:
        export_gif(mesh, outfile, rotation, elevation, quality)


    if exp_obj:
        obj_path = export_obj(mesh, outfile)
        transform(obj_path, outfile, rotation, elevation, quality, exp_png, make_gif, exp_obj = False)
        
    return mesh


def export_png(mesh, outfile, rotation, elevation, quality):
        scene = mesh.scene()

        rotation_matrix = transformations.rotation_matrix(
            -1 * rotation * math.pi / 180, [0, 0, 1], [0, 0, 0]
        )

        scene.apply_transform(rotation_matrix)

        elevation_matrix = transformations.rotation_matrix(
            -1 * elevation * math.pi / 180, [1, 0, 0], [0, 0, 0]
        )

        scene.apply_transform(elevation_matrix)

        png = scene.save_image(resolution=[res[quality], res[quality]], visible=False)

        output_path = outfile + ".png"

        with open(output_path, "wb") as f:
            f.write(png)
            f.close()

        print(f"created PNG: {output_path}")
        
        
def export_gif(mesh, outfile, rotation, elevation, quality):
        images = []
        rotations = np.linspace(0, 330, 12)
        for rotation in rotations:
            scene = mesh.scene()
            rotation_matrix = transformations.rotation_matrix(
                -1 * rotation * math.pi / 180, [0, 0, 1], [0, 0, 0]
            )
            scene.apply_transform(rotation_matrix)

            elevation_matrix = transformations.rotation_matrix(
                -1 * 45 * math.pi / 180, [1, 0, 0], [0, 0, 0]
            )
            scene.apply_transform(elevation_matrix)

            png = scene.save_image(resolution=[640, 640], visible=False)
            image = np.array(Image.open(BytesIO(png)))

            images.append(image)

        output_path = outfile + ".gif"
        imageio.mimsave(output_path, images)

        print(f"created GIF: {output_path}")


def export_obj(mesh, outfile):
        output_path = outfile + ".obj"
        mesh.export(output_path, file_type="obj")
        print(f"created OBJ: {output_path}")
        return output_path

def main():
    args = parse()
    args.png = True
    setup_dir(args.src, args.dest)

    if os.path.isfile(args.src):
        objfiles = [args.src]

    elif os.path.isdir(args.src):
        objfiles = [file for file in walk_dir(args.src)]

    else:
        raise ValueError("No valid source file type.")

    for i, file_path in enumerate(objfiles):
        path, file = os.path.split(file_path)
        outfile = os.path.normpath(os.path.join(args.dest, file)).split(".")[0]

        # transform(file_path, outfile, args.rot, args.ele, args.qual, i)

        p = multiprocessing.Process(
            target=transform,
            args=(
                file_path,
                outfile,
                args.rot,
                args.ele,
                args.qual,
                args.png,
                args.gif,
                args.obj
            ),
        )
        p.start()
        p.join(60)

        if p.is_alive():
            print("still running")
            p.terminate()
            p.join()

        print(f"Progress: {i / len(objfiles) * 100}")


if __name__ == "__main__":
    main()
