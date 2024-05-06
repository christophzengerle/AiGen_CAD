import sys

from config.configPcEncoder import ConfigPcEncoder
from AiGen_CAD.DeepCAD.dataset.pcEnc_dataset import get_dataloader
from AiGen_CAD.DeepCAD.trainer.trainerPCEncoder import TrainerPcEncoder

sys.path.append("..")
from cad2cad import decode_pc_zs
from utils import cycle


def encode(trainer, cfg):
    # load trained weights
    trainer.load_ckpt(cfg.ckpt)
    trainer.net.eval()

    # run PointCloud-Encoder
    return trainer.encode_pointcloud(cfg.pc_root)


def main():
    cfg = ConfigPcEncoder()
    print("data path:", cfg.pc_root)
    agent = TrainerPcEncoder(cfg)

    if cfg.exec == "train":
        # load from checkpoint if provided
        if cfg.cont:
            agent.load_ckpt(cfg.ckpt)
            
        # create dataloader
        train_loader = get_dataloader("train", cfg, noise=cfg.noise)
        val_loader = get_dataloader("validation", cfg)
        # val_loader = cycle(val_loader)

        # train
        agent.train(train_loader, val_loader)

    elif cfg.exec == "test":
        agent.load_ckpt(cfg.ckpt)
        agent.net.eval()
        
        # create dataloader
        test_loader = get_dataloader("test", cfg)
        
        agent.test(test_loader)
            
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
    main()
