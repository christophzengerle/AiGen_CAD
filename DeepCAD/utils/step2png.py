import argparse
import math
import multiprocessing
import os
import numpy as np
import imageio
from PIL import Image
from io import BytesIO

# from pyvirtualdisplay import Display


import trimesh
from trimesh import transformations


# when using with docker create virtual display
# export DISPLAY=192.168.178.29:0.0 


# run command: python step2png.py --src ../data/cad_step/ --dest ../data/cad_png/ --fdepth 2 --qual medium

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="source folder", required=True)
    parser.add_argument("--fdepth", type=int, default=1, help="source folder depth")
    parser.add_argument("--dest", type=str, default="png_files", help="destination folder")
    parser.add_argument("--ele", type=int, default=45, help="camera elevation")
    parser.add_argument("--rot", type=int, default=135, help="camera rotation")
    parser.add_argument("--gif", type=bool, default=False, help="make gif")
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


# def setup_virtual_display():
#     display = Display(visible=0, size=(1336, 768))
#     display.start()
    

def transform(file_path, outfile, rotation, elevation, quality, idx, res, make_gif):
    # setup_virtual_display()
    print('start', file_path)
    mesh = trimesh.Trimesh(
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

    #print(f"Progress: {idx / len(objfiles) * 100}")
    with open(outfile, "wb") as f:
        f.write(png)
        f.close()
    
    print(f'******** wrote to {outfile} *************')

    if make_gif:
        images = []
        rotations = np.linspace(0, 330, 12)
        for rotation in rotations:
            scene = mesh.scene()
            rotation_matrix = transformations.rotation_matrix(-1 * rotation * math.pi / 180, [0, 0, 1], [0, 0, 0])
            scene.apply_transform(rotation_matrix)

            elevation_matrix = transformations.rotation_matrix(-1 * 45 * math.pi / 180, [1, 0, 0], [0, 0, 0])
            scene.apply_transform(elevation_matrix)

            png = scene.save_image(resolution=[640, 640], visible=False)
            image = np.array(Image.open(BytesIO(png)))

            images.append(image)

        imageio.mimsave(outfile.replace(".png", ".gif"), images)


def main():
    args = parse()
    setup_dir(args.src, args.dest)

    if args.fdepth == 1:
        objfiles = [
            os.path.join(args.src, file)
            for file in os.listdir(args.src)
            if file.endswith(".step")
        ]
    elif args.fdepth == 2:
        objfiles = [
            os.path.join(args.src, folder, file)
            for folder in list(os.walk(args.src))[0][1]
            for file in os.listdir(os.path.join(args.src, folder))
            if file.endswith(".step")
        ]
    else:
        raise Exception(f"fdepth of {args.fdepth} not implemented yet")

    res = {"high": 1200, "medium": 600, "low": 300}

    for i, file_path in enumerate(objfiles):
        path, file = os.path.split(file_path)
        outfile = os.path.join(args.dest, file).replace(".step", ".png")

        if args.fdepth == 2:
            folder = os.path.split(path)[-1]
            if not os.path.exists(os.path.join(args.dest, folder)):
                os.mkdir(os.path.join(args.dest, folder))
            outfile = os.path.join(args.dest, folder, file).replace(".step", ".png")

        # transform(file_path, outfile, args.rot, args.ele, args.qual, i)


        p = multiprocessing.Process(
            target=transform, args=(file_path, outfile, args.rot, args.ele, args.qual, i, res, args.gif)
        )
        p.start()
        p.join(60)

        if p.is_alive():
            print("still running")
            p.terminate()
            p.join()


if __name__ == '__main__':
    main()