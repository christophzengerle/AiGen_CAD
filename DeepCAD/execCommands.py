import os

# cmd = "python pc2cad.py --exec train --exp_name pcEncNoNoise --batch_size 150 --num_workers 1 -g 1"

cmd = "python pc2cad.py --exec test --mode dec --pc_root evaluation/eval_pc_cad_images/point_e/ \
    --exp_name pcEncNoNoise --ckpt ckpt_epoch20_num4096 --ae_exp_name pretrained --ae_ckpt ckpt_epoch1000 \
    --n_points 1024 --expSTEP --expPNG --expGIF -g 0"

os.system(cmd)