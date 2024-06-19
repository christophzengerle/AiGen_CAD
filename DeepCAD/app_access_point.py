import os
import random

import numpy as np
import trimesh
from config.configPC2CAD import ConfigPC2CAD
from flask import Flask, jsonify, make_response, request
from trainer.trainerPC2CAD import TrainerPC2CAD

app = Flask(__name__)

random.seed(0)
np.random.seed(0)


###############################################################################
#################### Configuration ############################################
###############################################################################

DEEPCAD_EXPERIMENT_NAME = "pc2cad_contDiffNums"
DEEPCAD_MODEL_CKPT = "latest"
LOAD_MODULAR_CKPT = False

POINTCLOUD_N_POINTS = 8096

EXPORT_SOURCE_PNG = True
EXPORT_STEP = True
EXPORT_PNG = True


cfg = ConfigPC2CAD()
cfg.model_dir = os.path.join(cfg.proj_dir, f"pc2cad/{DEEPCAD_EXPERIMENT_NAME}/model")
cfg.ckpt = DEEPCAD_MODEL_CKPT
cfg.n_points = POINTCLOUD_N_POINTS
cfg.load_modular_ckpt = LOAD_MODULAR_CKPT
cfg.pce_exp_name = "pcEncoder"
cfg.pce_ckpt = "latest"
cfg.ae_exp_name = "pretrained"
cfg.ae_ckpt = "ckpt_epoch1000"

cfg.expSourcePNG = EXPORT_SOURCE_PNG
cfg.expSTEP = EXPORT_STEP
cfg.expPNG = EXPORT_PNG

agent = TrainerPC2CAD(cfg)
print("Loading DeepCAD-Model...")
agent.load_ckpt()
print("Loading Finished!")


@app.route("/", methods=["GET", "POST"])
def init():
    return make_response("DeepCAD running...", 200)


@app.route("/Object2PointCloud", methods=["POST"])
def obj2pc():
    # obj_path, out_path
    data = request.json
    obj_path = data["obj_path"]
    out_path = data["out_path"]
    m = trimesh.load_mesh(obj_path)
    path = os.path.join(out_path, "deepCAD.ply")
    pc = trimesh.PointCloud(m.sample(POINTCLOUD_N_POINTS))
    pc.export(path, file_type="ply")
    response = {"path": path}

    return jsonify(response)


@app.route("/GenerateCAD", methods=["POST"])
def deepcad_pc2cad():
    # file_path, output_path
    data = request.json
    file_path = data["pc_path"]
    output_path = data["output_path"]

    agent.cfg.pc_root = file_path
    print("data path:", agent.cfg.pc_root)
    if output_path:
        agent.cfg.output = output_path
    agent.cfg.expPNG = True
    agent.cfg.expSTEP = True
    
    out_path = agent.pc2cad()
    if out_path:
        response = {"STEP_path": out_path}
        return jsonify(response)
    return "DeepCAD could not create CAD-Model", 400


@app.route("/STEP2Object", methods=["POST"])
def step2Obj():
    # obj_path, out_path
    data = request.json
    step_path = data["step_path"]
    out_path = data["out_path"]

    m = trimesh.Trimesh(
        **trimesh.interfaces.gmsh.load_gmsh(
            file_name=step_path,
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

    outfile = os.path.join(out_path, "deepCAD.obj")
    m.export(outfile, file_type="obj")

    response = {"obj_path": outfile}

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8092)
