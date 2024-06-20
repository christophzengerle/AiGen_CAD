import os

# using nohup to run process in background -> terminal can be closed
# nohup python ... &
# less nohup.out  -> view output


# cmd = "nohup python pc2cad.py --exec train --exp_name pc2cad_ReduceLRscheduler_8096_1000epochs --batch_size 590 \
#     --nr_epochs 1000 --noise --n_points 8096 -g 0 &"
#   --continue --load_modular_ckpt --pce_exp_name pcEncoder --pce_ckpt latest \
#  --ae_exp_name pretrained --ae_ckpt ckpt_epoch1000 &"


# cmd = "python ./evaluation/eval_testimages/sample_random_ply.py"

# os.system(cmd)


# cmd = "python pc2cad.py --exec inf --pc_root evaluation/eval_testimages/source \
#     --exp_name pc2cad_contDiffNums --ckpt latest --n_points 8096 --expSTEP --expPNG -g 0 --output ./results"


# COV - JSD
# cmd = "python pc2cad.py --exec eval --mode gen --num_worker 8  \
#         --exp_name pc2cad_contDiffNums --ckpt latest --n_points 8096 -g 0 --batch_size 32"


# cmd = "python pc2cad.py --exec eval --mode acc --num_worker 8  \
#         --exp_name pc2cad_LRscheduler_8096 --ckpt ckpt_epoch200_num8096 --n_points 8096 -g 0"
#         --load_modular_ckpt --pce_exp_name pcEncRandNoise100New --pce_ckpt latest \
#         --ae_exp_name pretrained --ae_ckpt ckpt_epoch1000"


cmd = "tensorboard --logdir proj_log/pc2cad/pc2cad_ReduceLRscheduler_8096_1000epochs/log --host 0.0.0.0"

os.system(cmd)
