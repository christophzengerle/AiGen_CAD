import cv2

from src.models.lrm_mesh import InstantMesh
import trimesh
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.utils.camera_util import get_circular_camera_poses, spherical_camera_pose, get_zero123plus_input_cameras


def random_camera_pose(radius=3.0):
    theta_x = np.random.uniform(0, 2 * np.pi)  # theta is the angle with the z-axis
    theta_y = np.random.uniform(0, 2 * np.pi)
    theta_z = np.random.uniform(0, 2 * np.pi)
    # theta_x = 0
    # theta_y = 0
    # theta_z = 0

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
        [1, 0, 0, 2],
        [0, 1.0, 0.0, 2],
        [0, 0, 1.0, 2],
        [0.0, 0.0, 0.0, 1.0],
    ])
    pose = np.dot(pose, pose2)
    pose = np.dot(rot_x, pose)
    pose = np.dot(rot_y, pose)
    pose = np.dot(rot_z, pose)
    # pose[3,3] = 0
    return pose, theta_x, theta_y, theta_z


model = InstantMesh()

m = trimesh.load("/usr/app/src/instantmesh/data/obj_files/00000007.obj")
m.vertices -= m.center_mass
device = torch.device('cuda')
model.init_flexicubes_geometry(device=device, fovy=49)

for i in range(10):
    azim = 0
    ele = 0
    radius = 0
    depth_np = 0
    normal_np = 0
    sum = 0
    while sum == 0:

        vert = torch.Tensor([m.vertices]).to(device)
        face = torch.Tensor([m.faces]).to(device)

        azim = np.random.uniform(0, 360)
        ele = np.random.uniform(250, 270)
        radius = np.random.uniform(2, 10)
        cam = spherical_camera_pose(82, ele, radius=3).unsqueeze(0).unsqueeze(0).to(device)
        print(cam)
        cam, t_x, t_y, t_z = random_camera_pose(3)
        cam = torch.Tensor(cam).unsqueeze(0).unsqueeze(0).to(device)
        print(cam)
        # cam = get_zero123plus_input_cameras()[:, 0, :].reshape(1, 1, 4, 4).to(device)
        # cam[:, :, 3, 3] = 0

        mask, hard_mask, tex_pos, depth, normal = model.render_mesh(mesh_v=vert, mesh_f=face, cam_mv=cam)

        sum = depth.sum().to('cpu').numpy().item()

        if sum != 0:

            depth_np = depth.squeeze().to('cpu').numpy()
            depth_np = depth_np - np.min(depth_np)
            depth_np = (depth_np / np.max(depth_np) * 255).astype(int)

            normal_np = normal.squeeze().to('cpu').numpy()
            normal_np = normal_np - np.min(normal_np)
            normal_np = (normal_np / np.max(normal_np) * 255).astype(int)

    cv2.imwrite(f"/usr/app/src/instantmesh/data/00000007_depth_{t_x:.2f}_{t_y:.2f}_{t_z:.2f}.png", depth_np)
    # cv2.imwrite(f"/usr/app/src/instantmesh/data/00000007_normal_{t_x:.2f}_{t_y:.2f}_{t_z:.2f}.png", normal_np)

    # cv2.imwrite(f"/usr/app/src/instantmesh/data/00000007_depth_{azim:.2f}_{ele:.2f}_{radius:.2f}.png", depth_np)