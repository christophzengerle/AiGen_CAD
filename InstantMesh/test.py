from src.models.lrm_mesh import InstantMesh
import trimesh
import numpy as np
import torch

model = InstantMesh()

m = trimesh.load("../data/mesh_files/00000007.obj")

device = torch.device('cuda')
model.init_flexicubes_geometry(device=device)
model.render_mesh(mesh_v=m.vertices, mesh_f=m.faces, cam_mv=np.identity(4))

sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/