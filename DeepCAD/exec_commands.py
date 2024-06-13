import os

# using nohup to run process in background -> terminal can be closed
# nohup python ... &
# less nohup.out  -> view output


# cmd = "python pcEnc.py --exec train --exp_name pcEncoder --ckpt ckpt_epoch150_num8192 --continue \
#     --batch_size 175 --num_workers 8 --nr_epochs 200 --noise --n_points 2048 \
#     -g 1"


# cmd = "python pcEnc.py --exec inf --mode rec --pc_root evaluation/eval_pc_cad_images/point_e \
#     --exp_name pcEncRandNoise100New --ckpt latest --ae_exp_name pretrained --ae_ckpt ckpt_epoch1000 \
#     --n_points 4096 --expSTEP --expPNG --expGIF -g 0"




# cmd = "python cad2cad.py --exec test --mode dec --exp_name pretrained --ckpt ckpt_epoch1000 --num_workers 1 \
#         --expSTEP --expPNG --expGIF -g 0 \
#          --z_path proj_log/pce/pcEncNoNoise/results/pcEncodings/ckpt_epoch20_num4096/00195139_2024-05-01-23-49-26/00195139.h5"


         

# cmd = "python pc2cad.py --exec train --exp_name pc2cad_final_8192 --batch_size 590 \
#     --nr_epochs 1000 --noise --n_points 8192 -g 1"
        #   --continue --load_modular_ckpt --pce_exp_name pcEncoder --pce_ckpt latest \
        #  --ae_exp_name pretrained --ae_ckpt ckpt_epoch1000 &"


# cmd = "python pc2cad.py --exec inf --pc_root evaluation/eval_testimages/point_e \
#     --exp_name pc2cad_contAEckpt100 --ckpt latest --n_points 4096 --expSTEP --expPNG --expGIF -g 0"


cmd = "python pc2cad.py --exec eval --mode acc --num_worker 8  \
        --exp_name pc2cad_contDiffNums --ckpt latest --n_points 8192 -g 1 "
#         --load_modular_ckpt --pce_exp_name pcEncRandNoise100New --pce_ckpt latest \
#         --ae_exp_name pretrained --ae_ckpt ckpt_epoch1000"



# cmd = "tensorboard --logdir proj_log/pc2cad/pc2cad_final_8192/log --host 0.0.0.0"

os.system(cmd)
