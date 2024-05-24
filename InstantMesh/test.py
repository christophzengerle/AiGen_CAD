from src.models.lrm_mesh import InstantMesh
import trimesh
import numpy as np
import torch

model = InstantMesh()

m = trimesh.load("../utils/data/mesh_files/00000007.obj")

device = torch.device('cuda')
model.init_flexicubes_geometry(device=device)
mask, hard_mask, tex_pos, depth, normal = model.render_mesh(mesh_v=m.vertices, mesh_f=m.faces, cam_mv=np.identity(4))
