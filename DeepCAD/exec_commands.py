import os

# cmd = "python pc2cad.py --exec train --exp_name pcEncoderRandomNoise --batch_size 150 --num_workers 1 \
#     --nr_epochs 50 --noise -g 0"

cmd = "python pc2cad.py --exec test --mode dec --pc_root evaluation/eval_pc_cad_images/source \
    --exp_name pcEncoderRandomNoise --ckpt latest --ae_exp_name pretrained --ae_ckpt ckpt_epoch1000 \
    --n_points 4096 --expSTEP --expPNG --expGIF -g 0"
    
    
# cmd = "python cad2cad.py --exec test --mode dec --exp_name pretrained --ckpt ckpt_epoch1000 --num_workers 1 \
#         --expSTEP --expPNG --expGIF -g 0 \
#          --z_path proj_log/pce/pcEncNoNoise/results/pcEncodings/ckpt_epoch20_num4096/00195139_2024-05-01-23-49-26/00195139.h5"

# cmd = "tensorboard --logdir proj_log/pce/pcEncoderRandomNoise/log --host 0.0.0.0"

os.system(cmd)
