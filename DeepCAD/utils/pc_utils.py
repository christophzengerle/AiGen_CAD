import numpy as np
from plyfile import PlyData, PlyElement


def read_ply(path, with_normal=False):
    with open(path, "rb") as f:
        plydata = PlyData.read(f)
        x = np.array(plydata["vertex"]["x"])
        y = np.array(plydata["vertex"]["y"])
        z = np.array(plydata["vertex"]["z"])
        vertex = np.stack([x, y, z], axis=1)
        if with_normal:
            nx = np.array(plydata["vertex"]["nx"])
            ny = np.array(plydata["vertex"]["ny"])
            nz = np.array(plydata["vertex"]["nz"])
            normals = np.stack([nx, ny, nz], axis=1)
    if with_normal:
        return np.concatenate([vertex, normals], axis=1)
    else:
        return vertex


def write_ply(points, filename, text=False):
    """input: Nx3, write points to filename as PLY format."""
    points = [
        (points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])
    ]
    vertex = np.array(points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    el = PlyElement.describe(vertex, "vertex", comments=["vertices"])
    with open(filename, mode="wb") as f:
        PlyData([el], text=text).write(f)
