import sys
sys.path.append("./utils")
from config import ConfigAE
from dataset.cad_dataset import get_dataloader
from trainer import TrainerAE

from utils.file_utils import cycle


def decode_pc_zs(pc_config):
    sys.argv[1:] = list(
        map(
            str,
            [
                "--exec",
                pc_config.exec,
                "--exp_name",
                pc_config.ae_exp_name,
                "--ckpt",
                pc_config.ae_ckpt,
                "--gpu_ids",
                pc_config.gpu_ids,
            ],
        )
    )
    cfg = ConfigAE()
    cfg.set_pc_decoder_configuration(pc_config)
    tr_agent = TrainerAE(cfg)
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.decode_zs(cfg)
    

def main():
    # create experiment cfg containing all hyperparameters
    cfg = ConfigAE()
    print("data path:", cfg.data_root)

    # create network and training agent
    tr_agent = TrainerAE(cfg)

    # execute mode
    if cfg.exec == "train":
        # load from checkpoint if provided
        if cfg.cont:
            tr_agent.load_ckpt(cfg.ckpt)

        # create dataloader
        train_loader = get_dataloader("train", cfg)
        val_loader = get_dataloader("validation", cfg)
        val_loader = cycle(val_loader)

        tr_agent.train(train_loader, val_loader)

    elif cfg.exec == "test":
        # load from checkpoint if provided
        tr_agent.load_ckpt(cfg.ckpt)
        tr_agent.net.eval()
        
        # create dataloader
        test_loader = get_dataloader("test", cfg)
        
        tr_agent.test(test_loader)

        
    elif cfg.exec == "inf":
        # load from checkpoint if provided
        tr_agent.load_ckpt(cfg.ckpt)
        tr_agent.net.eval()

        if cfg.mode == "rec":
            tr_agent.reconstruct_vecs(cfg)
        elif cfg.mode == "enc":
            tr_agent.encode_vecs(cfg)
        elif cfg.mode == "dec":
            tr_agent.decode_zs(cfg)
        else:
            raise ValueError("Invalid mode.")

    else:
        raise ValueError(
            "Invalid execution type. Please specify --exec 'train', 'test' or 'inf' mode"
        )


if __name__ == "__main__":
    main()
