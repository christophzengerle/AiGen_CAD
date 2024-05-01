import os

# cmd = "python pc2cad.py --exec train --exp_name pcEncNoNoise --batch_size 150 --num_workers 1 -g 1"

cmd = "python pc2cad.py --exec test --mode dec --pc_root evaluation/eval_pc_cad_images/source/ \
    --exp_name pcEncNoNoise --ckpt ckpt_epoch20_num4096 --ae_exp_name pretrained --ae_ckpt ckpt_epoch1000 \
    --n_points 4096 --expSTEP --expPNG --expGIF -g 0"
    
    
# cmd = "python cad2cad.py --exec test --mode dec --exp_name pretrained --ckpt ckpt_epoch1000 --num_workers 1 \
#         --expSTEP --expPNG --expGIF -g 0 \
#          --z_path proj_log/pce/pcEncNoNoise/results/pcEncodings/ckpt_epoch20_num4096/source_2024-05-01-18-21-41/00195139.h5"

os.system(cmd)