import sys

from config.configPcEncoder import ConfigPcEncoder
from dataset.pc_dataset import get_dataloader
from trainer.trainerPcEncoder import TrainerPcEncoder

sys.path.append("..")
from cad2cad import decode
from utils import cycle

def main():
    cfg = ConfigPcEncoder()
    print("data path:", cfg.data_root)
    agent = TrainerPcEncoder(cfg)

    if not cfg.test:
        # load from checkpoint if provided
        if cfg.cont:
            agent.load_ckpt(cfg.ckpt)
        # create dataloader
        train_loader = get_dataloader("train", cfg)
        val_loader = get_dataloader("validation", cfg)
        val_loader = cycle(val_loader)

        # train
        agent.train(train_loader, val_loader)


    else:
        # load trained weights
        agent.load_ckpt(cfg.ckpt)

        # run PointCloud-Encoder
        pc_encodings, encodings_path = agent.encode_pointcloud(cfg.pc_root)
        
        decode(encodings_path, zs=pc_encodings, exportSTEP=True)

if __name__ == '__main__':
    main()