import argparse
import multiprocessing
import os
import numpy as np
import sys
import trimesh
import cv2
import pyrender
import json

sys.path.append("../../../src")

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="source folder", required=True)
    parser.add_argument("--dest", type=str, default="png_files", help="destination folder")
    parser.add_argument("--ele", type=int, default=45, help="camera elevation")
    parser.add_argument("--rot", type=int, default=135, help="camera rotation")
    parser.add_argument("--gif", type=bool, default=False, help="make gif")
    parser.add_argument(
        "--res",
        type=int,
        default=512,
        help="camera rotation",
    )
    parser.add_argument("--split", type=str, help="train-test-split", required=True)
    args = parser.parse_args()
    return args


def setup_dir(source_folder, destination_folder):
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Path {source_folder} does not exist")

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        os.makedirs(os.path.join(destination_folder, "train"))
        os.makedirs(os.path.join(destination_folder, "test"))


def walk_dir(dir):
    file_list = []
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            file_list.append(os.path.normpath(os.path.join(subdir, file)))
    return file_list


def random_camera_pose(radius=2.0):
    theta_x = np.random.uniform(0, 2 * np.pi)  # theta is the angle with the z-axis
    theta_y = np.random.uniform(0, 2 * np.pi)
    theta_z = np.random.uniform(0, 2 * np.pi)

    rot_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0, 0],
        [np.sin(theta_z), np.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    rot_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x), 0],
        [0, np.sin(theta_x), np.cos(theta_x), 0],
        [0, 0, 0, 1]
    ])

    rot_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y), 0],
        [0, 0, 0, 1]
    ])

    pose = np.array([
        [0.0, -np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
        [1.0, 0.0, 0.0, 0],
        [0.0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    pose2 = np.array([
        [1, 0, 0, 0],
        [0, 1.0, 0.0, 0],
        [0, 0, 1.0, 2],
        [0.0, 0.0, 0.0, 1.0],
    ])
    pose = np.dot(pose, pose2)
    pose = np.dot(rot_x, pose)
    pose = np.dot(rot_y, pose)
    pose = np.dot(rot_z, pose)

    return pose


class CustomShaderCache():
    def __init__(self):
        self.program = None

    def get_program(self, vertex_shader, fragment_shader, geometry_shader=None, defines=None):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram("src/shaders/mesh.vert", "src/shaders/mesh.frag", defines=defines)
        return self.program


def transform(file_path, out_folder, res, num):
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

    else:
        raise ValueError("Invalid File-Type {}.".format(file_path.split(".")[-1]))

    camera_poses = []
    for idx in range(num):
        mesh = pyrender.Mesh.from_trimesh(m)
        scene = pyrender.Scene()
        mesh_node = scene.add(mesh)
        ren = pyrender.OffscreenRenderer(800, 800)
        # Create a camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

        # Generate a random camera pose
        camera_pose = random_camera_pose()

        # Add the camera to the scene with the specified pose
        camera_node = scene.add(camera, pose=camera_pose)

        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                                   innerConeAngle=np.pi / 16.0,
                                   outerConeAngle=np.pi / 6.0)
        light_node = scene.add(light, pose=camera_pose)

        # Render the scene
        color, depth = ren.render(scene, flags=2048)

        depth = (depth/np.max(depth)*255).astype(np.uint8)

        # cv2.imshow("depth", depth)
        # cv2.imshow("color", color)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite(os.path.join(out_folder, '%03d.png' % idx), color)
        cv2.imwrite(os.path.join(out_folder, '%03d_depth.png' % idx), depth)

        ren._renderer._program_cache = CustomShaderCache()

        normal, depth = ren.render(scene, flags=2048)

        cv2.imwrite(os.path.join(out_folder, '%03d_normal.png' % idx), normal)

        camera_poses.append(camera_pose[:3, :])

        # Remove the camera from the scene
        scene.remove_node(camera_node)
        scene.remove_node(light_node)

    with open(os.path.join(out_folder, "cameras.npz"), "wb") as f:
        np.savez(f, cam_poses=np.array(camera_poses))
        f.close()


def main():
    train_files = []
    val_files = []
    test_files = []
    failed_files = []
    args = parse()
    setup_dir(args.src, args.dest)
    
    with open(args.split, "r") as f:
        train_test_split = f.read()
        f.close()

    train = train_test_split["train"]
    val = train_test_split["validation"]
    test = train_test_split["test"]

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
        out_folder = os.path.join(args.dest, file).split('.')[0]
        
        train_files.append(os.path.abspath(out_folder))

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # transform(file_path, outfile, args.rot, args.ele, args.qual, i)

        p = multiprocessing.Process(
            target=transform, args=(file_path, out_folder, args.res, 32)
        )
        p.start()
        p.join(60)

        if p.is_alive():
            print("still running")
            p.terminate()
            p.join()
            failed_files.append(os.path.abspath(out_folder))

        print(f"Progress: {(i+1) / len(objfiles) * 100}")

    file_path_dict = {"good_objs": train_files, "val_objs": val_files, "test_objs": test_files}
    with open(os.path.join(args.dest, "valid_paths.json"), "w") as f:
        json.dump(file_path_dict, f)
    with open(os.path.join(args.dest, "failed_files.json"), "w") as f:
        json.dump(file_path_dict, f)


if __name__ == '__main__':
    main()
