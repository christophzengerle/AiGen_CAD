import sys

from config.configPcEncoder import ConfigPcEncoder
from dataset.pc_dataset import get_dataloader
from trainer.trainerPcEncoder import TrainerPcEncoder

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
        train_loader = get_dataloader("train", cfg)
        val_loader = get_dataloader("validation", cfg)
        # val_loader = cycle(val_loader)

        # train
        agent.train(train_loader, val_loader)

    elif cfg.exec == "test":
        if cfg.mode == "enc":
            cfg.zs = encode(agent, cfg)

        elif cfg.mode == "dec":
            cfg.zs = encode(agent, cfg)
            decode_pc_zs(cfg)

        else:
            raise ValueError(
                "Invalid execution mode. Please specify --mode 'enc' or 'dec' mode"
            )

    else:
        raise ValueError(
            "Invalid execution type. Please specify --exec 'train' or 'test' mode"
        )


if __name__ == "__main__":
    main()
