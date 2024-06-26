import random
import numpy as np
import sys
sys.path.append("./utils")
from config.configPCEncoder import ConfigPCEncoder
from dataset.pcEnc_dataset import get_dataloader
from trainer.trainerPCEncoder import TrainerPCEncoder

from cad2cad import decode_pc_zs


def encode(trainer, cfg):
    # load trained weights
    trainer.load_ckpt(cfg.ckpt)
    trainer.net.eval()

    # run PointCloud-Encoder
    return trainer.encode_pointcloud(cfg.pc_root)


def main():
    cfg = ConfigPCEncoder()
    print("data path:", cfg.pc_root)
    agent = TrainerPCEncoder(cfg)

    if cfg.exec == "train":
        # load from checkpoint if provided
        if cfg.cont:
            agent.load_ckpt(cfg.ckpt)
            
        # create dataloader
        train_loader = get_dataloader("train", cfg, noise=cfg.noise)
        val_loader = get_dataloader("validation", cfg)
        test_loader = get_dataloader("test", cfg)

        # train
        agent.train(train_loader, val_loader, test_loader)

    elif cfg.exec == "eval":
        agent.load_ckpt(cfg.ckpt)
        
        # create dataloader
        test_loader = get_dataloader("test", cfg)
        
        agent.eval_model_acc(test_loader)
            
    elif cfg.exec == "inf":
        cfg.zs = encode(agent, cfg)
        if cfg.mode == "enc":
            pass
        elif cfg.mode == "rec":
            decode_pc_zs(cfg)
        else:
            raise ValueError(
                "Invalid execution mode. Please specify --mode 'enc' or 'rec' mode"
            )

    else:
        raise ValueError(
            "Invalid execution type. Please specify --exec 'train', 'test' or 'inf' mode"
        )


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    main()
