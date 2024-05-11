import os

# cmd = "python pcEnc.py --exec train --exp_name pcEncRandNoise100New \
#     --batch_size 150 --num_workers 1 --nr_epochs 100 --noise -g 0"


cmd = "python pc2cad.py --exec train --exp_name pc2cad_test100 --num_workers 8 --batch_size 175 \
    --nr_epochs 100 --noise -g 1  \
        --continue --load_modular_ckpt --pce_exp_name pcEncRandNoise100New --pce_ckpt latest \
       --ae_exp_name pretrained --ae_ckpt ckpt_epoch1000"


# cmd = "python pcEnc.py --exec inf --mode rec --pc_root evaluation/eval_pc_cad_images/point_e \
#     --exp_name pcEncRandNoise100New --ckpt latest --ae_exp_name pretrained --ae_ckpt ckpt_epoch1000 \
#     --n_points 4096 --expSTEP --expPNG --expGIF -g 0"

# cmd = "python pc2cad.py --exec inf --pc_root evaluation/eval_testimages/point_e/00001817.ply \
#     --exp_name pc2cad_Exp --ckpt latest --n_points 4096 --expSTEP --expPNG --expGIF -g 0"


# cmd = "python pc2cad.py --exec test --mode gen --num_worker 1  \
#         --exp_name pc2cad_Exp --ckpt latest --n_points 4096 -g 0 \
#         --load_modular_ckpt --pce_exp_name pcEncRandNoise100New --pce_ckpt latest \
#         --ae_exp_name pretrained --ae_ckpt ckpt_epoch1000"


# cmd = "python cad2cad.py --exec test --mode dec --exp_name pretrained --ckpt ckpt_epoch1000 --num_workers 1 \
#         --expSTEP --expPNG --expGIF -g 0 \
#          --z_path proj_log/pce/pcEncNoNoise/results/pcEncodings/ckpt_epoch20_num4096/00195139_2024-05-01-23-49-26/00195139.h5"

# cmd = "tensorboard --logdir proj_log/pc2cad/pc2cad_test/log --host 0.0.0.0"

os.system(cmd)
