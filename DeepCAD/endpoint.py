from DeepCAD.config.configPC2CAD import ConfigPC2CAD
from DeepCAD.trainer.trainerPC2CAD import TrainerPC2CAD

EXPERIMENT_NAME_DEEPCAD = "pc2cad_8192"


def endpoint(file_path, output_path = None):
    cfg = ConfigPC2CAD()
    print("data path:", cfg.pc_root)
    cfg.pc_root = file_path
    if output_path:
        cfg.output = output_path
    cfg.expPNG = True
    cfg.expSTEP = True
    cfg.model_dir = f"proj_log/pc2cad/{EXPERIMENT_NAME_DEEPCAD}/model"
    cfg.ckpt = "latest"
    agent = TrainerPC2CAD(cfg)
    agent.load_ckpt()
    out_path = agent.pc2cad()
    return out_path
