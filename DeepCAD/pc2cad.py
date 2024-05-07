import sys

from config.configPC2CAD import ConfigPC2CAD
from dataset.pc2cad_dataset import get_dataloader
from trainer.trainerPC2CAD import TrainerPC2CAD

sys.path.append("..")


def main():
    cfg = ConfigPC2CAD()
    print("data path:", cfg.pc_root)
    agent = TrainerPC2CAD(cfg)

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

    elif cfg.exec == "test":
        agent.load_ckpt(cfg.ckpt)
        agent.net.eval()

        # create dataloader
        test_loader = get_dataloader("test", cfg)

        agent.test(test_loader)

    elif cfg.exec == "inf":
        agent.load_ckpt(cfg.ckpt)
        agent.net.eval()

        agent.pc2cad()

    else:
        raise ValueError(
            "Invalid execution type. Please specify --exec 'train', 'test' or 'inf' mode"
        )


if __name__ == "__main__":
    main()
