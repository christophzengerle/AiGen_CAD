import os
import trimesh
from config.configPC2CAD import ConfigPC2CAD
from trainer.trainerPC2CAD import TrainerPC2CAD

from flask import Flask, request, make_response, jsonify

app = Flask(__name__)

print('Loading DeepCAD-Model...')
DEEPCAD_EXPERIMENT_NAME = "pc2cad_Exp"
DEEPCAD_MODEL_CKPT = "latest"
POINTCLOUD_N_POINTS = 4096

cfg = ConfigPC2CAD()
cfg.model_dir = os.path.join(cfg.proj_dir, f"pc2cad/{DEEPCAD_EXPERIMENT_NAME}/model")
cfg.ckpt = DEEPCAD_MODEL_CKPT
cfg.n_points = POINTCLOUD_N_POINTS
cfg.load_modular_ckpt = True
cfg.pce_exp_name = 'pcEncoder'
cfg.pce_ckpt = "latest"
cfg.ae_exp_name = "pretrained"
cfg.ae_ckpt = "ckpt_epoch1000"

agent = TrainerPC2CAD(cfg)
agent.load_ckpt()
print('Loading Finished!')


@app.route('/',methods=['GET','POST'])
def init():
    return make_response("DeepCAD running...", 200)



@app.route('/Object2PointCloud', methods=['POST'])
def obj2pc():
    # obj_path, out_path
    data = request.json
    obj_path = data['obj_path']
    out_path = data['out_path']
    m = trimesh.load_mesh(obj_path)
    path = os.path.join(out_path, "instantMesh.ply")
    pc = trimesh.PointCloud(m.sample(POINTCLOUD_N_POINTS))
    pc.export(path, file_type='ply')
    response = {
        'path' : path
    }
    
    return jsonify(response)

@app.route('/GenerateCAD', methods=['POST'])
def deepcad_pc2cad():
    # file_path, output_path
    data = request.json
    file_path = data['pc_path']
    output_path = data['output_path']
    
    agent.cfg.pc_root = file_path
    print("data path:", agent.cfg.pc_root)
    if output_path:
        agent.cfg.output = output_path
    agent.cfg.expPNG = True
    agent.cfg.expSTEP = True
    try:
        out_path = agent.pc2cad()
        response = {
            'STEP_path' : out_path
        }
        return jsonify(response)
    except:
        return "DeepCAD could not create CAD-Model", 400
        # make_response("DeepCAD could not create CAD-Model!", 500)
    

@app.route('/STEP2Object', methods=['POST'])
def step2Obj():
    # obj_path, out_path
    data = request.json
    step_path = data['step_path']
    out_path = data['out_path']
    
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
    m.export(outfile, file_type='obj')
 
    response = {
        'obj_path' : outfile
    }
    
    return jsonify(response)
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
