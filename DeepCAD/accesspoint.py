import os
import random
import sys
sys.path.append("./utils")

import numpy as np
import trimesh
from trimesh import transformations
from config.configPC2CAD import ConfigPC2CAD
from flask import Flask, jsonify, make_response, request
from trainer.trainerPC2CAD import TrainerPC2CAD
from utils.obj_utils import create_mesh_from_step
from utils.pc_utils import write_ply
import math

app = Flask(__name__)

random.seed(0)
np.random.seed(0)


###############################################################################
#################### Configuration ############################################
###############################################################################

# DEEPCAD_EXPERIMENT_NAME = "pc2cad_contDiffNums"
# DEEPCAD_MODEL_CKPT = "latest"

DEEPCAD_EXPERIMENT_NAME = "pc2cad_FinalTransform_8096_1000epochs"
DEEPCAD_MODEL_CKPT = "ckpt_epoch750_num8096"

# DEEPCAD_EXPERIMENT_NAME = "pc2cad_MoreTransform_8096_1000epochs"
# DEEPCAD_MODEL_CKPT = "ckpt_epoch350_num8096"

# DEEPCAD_EXPERIMENT_NAME = "pc2cad_ReduceLRscheduler_8096_1000epochs"
# DEEPCAD_MODEL_CKPT = "ckpt_epoch300_num8096"



LOAD_MODULAR_CKPT = False

POINTCLOUD_N_POINTS = 8096

EXPORT_SOURCE_PNG = True
EXPORT_STEP = True
EXPORT_PNG = True

GPU_IDS = "0"


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

cfg.gpu_ids = GPU_IDS
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_ids)

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
    pc.vertices[:, [2, 1]] = pc.vertices[:, [1, 2]]    
    pc = pc.vertices
    # swap y and z axis
    write_ply(pc, path)
    # pc.export(path, file_type="ply")
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

    m = create_mesh_from_step(step_path)

    outfile = os.path.join(out_path, "deepCAD.obj")
    m.export(outfile, file_type="obj")

    response = {"obj_path": outfile}

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8092)
