import os
import sys

import h5py
import numpy as np
import torch
from cadlib.macro import EOS_IDX
from config import ConfigAE
from dataset.cad_dataset import get_dataloader
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.DataExchange import write_step_file
from tqdm import tqdm
from trainer import TrainerAE
from utils import ensure_dir

sys.path.append("..")
from cadlib.visualize import vec2CADsolid
import utils.step2png
from utils.step2png import transform
from utils.file_utils import walk_dir


# define different modes
def reconstruct(cfg, tr_agent):
    # create dataloader
    test_loader = get_dataloader("test", cfg)
    print("Total number of test data:", len(test_loader))

    if cfg.outputs is None:
        cfg.outputs = "{}/results/test_{}".format(cfg.exp_dir, cfg.ckpt)
    ensure_dir(cfg.outputs)

    # evaluate
    pbar = tqdm(test_loader)
    for i, data in enumerate(pbar):
        batch_size = data["command"].shape[0]
        commands = data["command"]
        args = data["args"]
        gt_vec = (
            torch.cat([commands.unsqueeze(-1), args], dim=-1)
            .squeeze(1)
            .detach()
            .cpu()
            .numpy()
        )
        commands_ = gt_vec[:, :, 0]
        with torch.no_grad():
            outputs, _ = tr_agent.forward(data)
            batch_out_vec = tr_agent.logits2vec(outputs)

        for j in range(batch_size):
            out_vec = batch_out_vec[j]
            seq_len = commands_[j].tolist().index(EOS_IDX)

            data_id = data["id"][j].split("/")[-1]

            save_path = os.path.join(cfg.outputs, "{}_vec.h5".format(data_id))
            with h5py.File(save_path, "w") as fp:
                fp.create_dataset("out_vec", data=out_vec[:seq_len], dtype=np.int)
                fp.create_dataset("gt_vec", data=gt_vec[j][:seq_len], dtype=np.int)


def encode(cfg, tr_agent):
    # create dataloader
    # save_dir = "{}/results".format(cfg.exp_dir)
    # ensure_dir(save_dir)
    # save_path = os.path.join(save_dir, "all_zs_ckpt{}.h5".format(cfg.ckpt))

    save_dir = "./data"
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, "all_zs.h5")

    # fp = h5py.File(save_path, "w")
    fp = h5py.File(save_path, "w")

    for phase in ["train", "validation", "test"]:
        train_loader = get_dataloader(phase, cfg, shuffle=False)

        # encode
        all_zs = []
        pbar = tqdm(train_loader)
        for i, data in enumerate(pbar):
            with torch.no_grad():
                z = tr_agent.encode(data, is_batch=True)
                z = z.detach().cpu().numpy()[:, 0, :]
                all_zs.append(z)
        all_zs = np.concatenate(all_zs, axis=0)
        print(all_zs.shape)
        fp.create_dataset("{}_zs".format(phase), data=all_zs)
    fp.close()



def decode(cfg):
    # create network and training agent
    tr_agent = TrainerAE(cfg)

    # load from checkpoint if provided
    tr_agent.load_ckpt(cfg.ckpt)
    tr_agent.net.eval()

    # load latent zs
    with h5py.File(cfg.z_path, 'r') as fp:
        zs = fp['zs'][:]
    save_dir = cfg.z_path.split('.')[0] + '_dec'
    ensure_dir(save_dir)

    # decode
    for i in range(0, len(zs), cfg.batch_size):
        with torch.no_grad():
            batch_z = torch.tensor(zs[i:i+cfg.batch_size], dtype=torch.float32).unsqueeze(1)
            batch_z = batch_z.cuda()
            outputs = tr_agent.decode(batch_z)
            batch_out_vec = tr_agent.logits2vec(outputs)

        for j in range(len(batch_z)):
            out_vec = batch_out_vec[j]
            out_command = out_vec[:, 0]
            seq_len = out_command.tolist().index(EOS_IDX)

            save_path = os.path.join(save_dir, '{}.h5'.format(i + j))
            with h5py.File(save_path, 'w') as fp:
                fp.create_dataset('out_vec', data=out_vec[:seq_len], dtype=np.int)



def inf_decode(cfg, tr_agent):
    zs = []
    save_paths = []
    batch_size = cfg.batch_size
    
    if not hasattr(cfg, 'zs'):
        if cfg.z_path:
            if os.path.isfile(cfg.z_path):
                if cfg.z_path.endswith(".h5") and not cfg.z_path.endswith("_dec.h5"):
                    with h5py.File(cfg.z_path, "r") as fp:
                        zs.append(fp["zs"][:])
                    save_paths.append(cfg.z_path)
                else:
                    raise ValueError("Invalid file format")

            elif os.path.isdir(cfg.z_path):
                for file in walk_dir(cfg.z_path):
                    if file.endswith(".h5") and not file.endswith("_dec.h5"):
                        with h5py.File(file, "r") as fp:
                            zs.append(fp["zs"][:])
                        save_paths.append(file)

            else:
                raise ValueError("Invalid path")
        else:
            raise ValueError("No zs provided.")

    else:
        zs = cfg.zs["zs"]
        save_paths = cfg.zs["z_paths"]

    # decode
    for i in range(0, len(zs), batch_size):
        with torch.no_grad():
            batch_z = torch.tensor(
                np.concatenate(zs[i : i + batch_size]), dtype=torch.float32
            ).unsqueeze(1)
            batch_z = batch_z.cuda()
            outputs = tr_agent.decode(batch_z)
            batch_out_vec = tr_agent.logits2vec(outputs)

        for j in range(len(batch_z)):
            print("\n*** File: " + save_paths[i + j].split("/")[-1] + " ***\n")
            out_vec = batch_out_vec[j]
            out_command = out_vec[:, 0]
            seq_len = out_command.tolist().index(EOS_IDX)
            out_vec = out_vec[:seq_len]
            
            
            save_path = save_paths[i + j].split(".")[0]
            
            
            out_shape = None
            is_valid_BRep = False
            try:
                out_shape = vec2CADsolid(out_vec)
                is_valid_BRep = True
            except Exception as e:
                print(f"Creation of CAD-Solid for file {save_path} failed.\n" + str(e.with_traceback))

                # check generated CAD-Shape 
                # if invalid -> generate again
                for cnt_retries in range(1, cfg.n_checkBrep_retries + 1):
                    print(f"Trying to create a new CAD-Solid. Attempt {cnt_retries}/{cfg.n_checkBrep_retries}")
                    # print(batch_z.shape, batch_z[j].shape)
                    new_batch_output = tr_agent.decode(batch_z[j].unsqueeze(0))
                    out_batch_vec = tr_agent.logits2vec(new_batch_output)
                    out_vec = out_batch_vec.squeeze(0)
                    
                    out_command = out_vec[:, 0]
                    seq_len = out_command.tolist().index(EOS_IDX)
                    out_vec = out_vec[:seq_len]
                    
                    try:
                        out_shape = vec2CADsolid(out_vec)
                        analyzer = BRepCheck_Analyzer(out_shape)
                        if analyzer.IsValid():
                            print("Valid BRep-Model detected.")
                            is_valid_BRep = True
                            break
                        else:
                            print("invalid BRep-Model detected.")
                            continue
                            
                    except Exception as e:
                        print(f"Creation of CAD-Solid for file {save_path} failed.\n" + str(e.with_traceback))
                        continue
                       
                                        
            if not is_valid_BRep:
                print('Could not create valid BRep-Model!')
                continue
            
            
            
            save_path_vec = save_path + "_dec.h5"
            with h5py.File(save_path_vec, "w") as fp:
                fp.create_dataset("out_vec", data=out_vec, dtype=np.int32)
                
                
            step_save_path = save_path + "_dec.step"
            if cfg.expSTEP:
                try:
                    create_step_file(out_shape, step_save_path)                     
                except Exception as e:
                    print(str(e.with_traceback))
                    continue
                
                
            if cfg.expPNG or cfg.expGIF:
                try:
                    png_path = step_save_path.split('.')[0]
                    if step_file_exists(step_save_path):
                        transform(step_save_path, png_path, 135, 45, "medium", exp_png=cfg.expPNG, make_gif=cfg.expGIF)
                        print(f"Image-Output for {png_path} created.")
                    else:
                        print(f'no .STEP-File for {step_save_path.split("/")[-1]} found.\nTrying to create .STEP-File') 
                        try:
                            create_step_file(out_shape, step_save_path)                     
                        except Exception as e:
                            raise Exception(str(e.with_traceback))    
                except Exception as e:
                    print(
                        f"Creation of Image-Output for {save_paths[i + j].split('/')[-1]} failed.\n"
                        + str(e.with_traceback)
                    )
                    continue
                
                
def step_file_exists(path):
    return os.path.isfile(path)
                
def create_step_file(out_shape, path):
    try:
        if not step_file_exists(path):
            write_step_file(out_shape, path)
            print("{} created.".format(path.split("/")[-1]))
            
        else:
            print(f".STEP-File {path.split('/')[-1]} already exists.")
        
    except Exception as e:
        raise Exception(f"Creation of .STEP-File for {path.split('/')[-1]} failed.\n"
                + str(e.with_traceback))
    
    

                
