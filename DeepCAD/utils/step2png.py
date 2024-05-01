import argparse
import math
import multiprocessing
import os
import numpy as np
import imageio
from PIL import Image
from io import BytesIO

from pyvirtualdisplay import Display


import trimesh
from trimesh import transformations


res = {"high": 1200, "medium": 600, "low": 300}

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
        
        
def setup_virtual_display():
    display = Display(visible=0)
    display.start()


    

def transform(file_path, outfile, rotation, elevation, quality, exp_png=True, make_gif=False):    
    print('start', file_path)
    setup_virtual_display()
    if file_path.endswith(".ply"):
        mesh = trimesh.load_mesh(file_path)
    elif file_path.endswith(".step"):
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
    
    else:
        raise ValueError("Invalid File-Type {}.".format(file_path.split(".")[-1]))  
        
    if exp_png:
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

        output_path = outfile + ".gif"
        imageio.mimsave(output_path, images)
    
    print(f'******** wrote to {outfile} *************')        


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
        raise NotImplementedError(f"fdepth of {args.fdepth} not implemented yet")

    for i, file_path in enumerate(objfiles):
        path, file = os.path.split(file_path)
        outfile = os.path.join(args.dest, file).split('.')[0]

        if args.fdepth == 2:
            folder = os.path.split(path)[-1]
            if not os.path.exists(os.path.join(args.dest, folder)):
                os.mkdir(os.path.join(args.dest, folder))
            outfile = os.path.join(args.dest, folder, file).split('.')[0]

        # transform(file_path, outfile, args.rot, args.ele, args.qual, i)


        p = multiprocessing.Process(
            target=transform, args=(file_path, outfile, args.rot, args.ele, args.qual, True, args.gif)
        )
        p.start()
        p.join(60)

        if p.is_alive():
            print("still running")
            p.terminate()
            p.join()
            
        print(f"Progress: {i / len(objfiles) * 100}")

if __name__ == '__main__':
    main()