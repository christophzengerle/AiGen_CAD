import os

from config.configPC2CAD import ConfigPC2CAD
from trainer.trainerPC2CAD import TrainerPC2CAD

from flask import Flask, request, make_response, jsonify

app = Flask(__name__)

print('Loading DeepCAD-Model...')
DEEPCAD_EXPERIMENT_NAME = "pc2cad_8192"
DEEPCAD_MODEL_CKPT = "latest"
cfg = ConfigPC2CAD()
cfg.model_dir = os.path.join(cfg.proj_dir, f"pc2cad/{DEEPCAD_EXPERIMENT_NAME}/model")
cfg.ckpt = DEEPCAD_MODEL_CKPT
agent = TrainerPC2CAD(cfg)
agent.load_ckpt()
print('Loading Finished!')


@app.route('/',methods=['GET','POST'])
def init():
    return make_response("DeepCAD running...", 200)

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
    out_path = agent.pc2cad()
    
    response = {
        'STEP_path' : out_path
    }
    
    return jsonify(response)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
