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
            agent.load_ckpt()

        # create dataloader
        train_loader = get_dataloader("train", cfg, noise=cfg.noise)
        val_loader = get_dataloader("validation", cfg)
        test_loader = get_dataloader("test", cfg)

        # train
        agent.train(train_loader, val_loader, test_loader)

    elif cfg.exec == "eval":
        agent.load_ckpt()
        # create dataloader
        test_loader = get_dataloader("test", cfg)

        if cfg.mode == "acc":
            agent.eval_model_acc(test_loader)

        elif cfg.mode == "cd":
            agent.eval_model_chamfer_dist(test_loader)

        elif cfg.mode == "gen":
            agent.eval_model_cov_mmd_jsd(test_loader)

        else:
            raise ValueError(
                "Invalid execution mode. Please specify --mode 'enc' or 'rec' mode"
            )

    elif cfg.exec == "inf":
        agent.load_ckpt()
        agent.pc2cad()

    else:
        raise ValueError(
            "Invalid execution type. Please specify --exec 'train', 'test' or 'inf' mode"
        )


if __name__ == "__main__":
    main()
