import argparse
import multiprocessing
import os
import numpy as np
import sys
import trimesh
import cv2
import json
sys.path.append("/usr/app/src/instantmesh/")
from src.models.lrm_mesh import InstantMesh
from src.utils.camera_util import (
    FOV_to_intrinsics,
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
    spherical_camera_pose
)
import torch
import numpy as np


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="source folder", required=True)
    parser.add_argument("--dest", type=str, default="png_files", help="destination folder")
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


def walk_dir(dir):
    file_list = []
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            file_list.append(os.path.normpath(os.path.join(subdir, file)))
    return file_list


def get_render_cameras(batch_size=1, radius=0.2):
    """
    Get the rendering camera parameters.
    """
    azim = np.random.uniform(0, 360)
    ele = np.random.uniform(0, 360)
    radius = radius
    c2ws = spherical_camera_pose(azimuths=azim, elevations=ele, radius=radius)

    cameras = torch.linalg.inv(c2ws)
    cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return cameras


def transform(model, mesh, out_folder, res, num, obj_with_no_mass):
    # print('filepath', file_path)
    m = mesh
    
    device = torch.device('cuda')
    
    camera_poses = []
    for idx in range(num):
        render_cameras = get_render_cameras()
        cam_mv = render_cameras
        camera_poses.append(cam_mv[:3, :])        
        
        vert = torch.Tensor([m.vertices]).to(device)
        face = torch.Tensor([m.faces]).to(device)

        mask, hard_mask, tex_pos, depth, normal = model.render_mesh(mesh_v=vert, mesh_f=face, cam_mv=cam_mv.to(device), render_size=res)

        depth_np = depth.squeeze().to('cpu').numpy()
        depth_np = depth_np - np.min(depth_np)
        depth_np = (depth_np / np.max(depth_np) * 255).astype(int)

        normal_np = normal.squeeze().to('cpu').numpy()
        normal_np = normal_np - np.min(normal_np)
        normal_np = (normal_np / np.max(normal_np) * 255).astype(int)

        mask_np = mask.squeeze().to('cpu').numpy()
        mask_np = mask_np - np.min(mask_np)
        mask_np = (mask_np / np.max(mask_np) * 255).astype(int)

        hard_mask_np = hard_mask.squeeze().to('cpu').numpy()
        hard_mask_np = hard_mask_np - np.min(hard_mask_np)
        hard_mask_np = (hard_mask_np / np.max(hard_mask_np) * 255).astype(int)

        tex_pos_np = tex_pos[0].squeeze().to('cpu').numpy()
        tex_pos_np = tex_pos_np - np.min(tex_pos_np)
        tex_pos_np = (tex_pos_np / np.max(tex_pos_np) * 255).astype(int)

        cv2.imwrite(os.path.join(out_folder, '%03d.png' % idx), normal_np)
        cv2.imwrite(os.path.join(out_folder, '%03d_depth.png' % idx), depth_np)
        cv2.imwrite(os.path.join(out_folder, '%03d_normal.png' % idx), normal_np)

    with open(os.path.join(out_folder, "cameras.npz"), "wb") as f:
        np.savez(f, cam_poses=np.array(camera_poses))
        f.close()


def main():
    model = InstantMesh()
    device = torch.device('cuda')
    model.init_flexicubes_geometry(device=device, fovy=49)
    train_files = []
    val_files = []
    test_files = []
    failed_files = []
    obj_with_no_mass = []
    args = parse()
    setup_dir(args.src, args.dest)

    with open(args.split, "r") as f:
        train_test_split = json.load(f)
        f.close()

    train = train_test_split["train"]
    val = train_test_split["validation"]
    test = train_test_split["test"]

    if os.path.isfile(args.src):
        if args.src.endswith(".obj"):
            objfiles = [args.src]

    elif os.path.isdir(args.src):
        objfiles = [
            file
            for file in walk_dir(args.src)
            if file.endswith(".obj")
        ]
    
    else:
        raise ValueError("No valid source file type.")

    for i, file_path in enumerate(objfiles):
        path, file = os.path.split(file_path)
        out_folder = os.path.join(args.dest, file).split('.')[0]
        file_id = os.path.join(os.path.split(path)[-1], file.replace(".obj", ""))

        m = trimesh.load_mesh(file_path)
        try:
            m.vertices -= m.center_mass
        except:
            obj_with_no_mass.append(file_path)
            print("Without mass: ", file_path)
            raise Exception

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # transform(file_path, outfile, args.rot, args.ele, args.qual, i)
        transform(model, m, out_folder, args.res, 32, obj_with_no_mass)
        
        if file_id in train:
            train_files.append(os.path.abspath(out_folder))
        elif file_id in val:
            val_files.append(os.path.abspath(out_folder))
        elif file_id in test:
            test_files.append(os.path.abspath(out_folder))

        if i%1000 == 0:
            print(f"File: {file_path}")
            print(f"Progress: {i}/{len(objfiles)} = {(i+1) / len(objfiles) * 100}")

    file_path_dict = {"good_objs": train_files, "val_objs": val_files, "test_objs": test_files, "failed_objs": failed_files}
    with open(os.path.join(args.dest, "valid_paths.json"), "w") as f:
        json.dump(file_path_dict, f)
        
    with open(os.path.join(args.dest, "no_mass.json"), "w") as f:
        json.dump(obj_with_no_mass, f)


if __name__ == '__main__':
    main()
