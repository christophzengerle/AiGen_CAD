import os

# using nohup to run process in background -> terminal can be closed
# nohup python ... &
# less nohup.out  -> view output


# cmd = "nohup python pc2cad.py --exec train --exp_name pcEncFinalTransform_8096_500epochs --batch_size 590 \
#     --nr_epochs 500 --noise --n_points 8096 -g 1 \
#   --continue \
#           --load_modular_ckpt --pce_exp_name pcEncoder_Transformation --pce_ckpt ckpt_epoch500_num8096 \
#         --ae_exp_name pretrained --ae_ckpt ckpt_epoch1000 &"

# cmd = "nohup python pcEnc.py --exec train --exp_name pcEncoder_Transformation --batch_size 620 \
#     --num_workers 8 --nr_epochs 1000 --noise --n_points 8096 -g 1 &"


# cmd = "python ./evaluation/eval_testimages/sample_random_ply.py"

# os.system(cmd)


# cmd = "python pc2cad.py --exec inf --pc_root evaluation/eval_testimages/source/ \
#     --exp_name pc2cad_Transformations_8096_1000epochs --ckpt ckpt_epoch200_num8096 \
#      --n_points 8096 --expSTEP --expPNG -g 1 --output ./results" \
#     --load_modular_ckpt --pce_exp_name pcEncRandNoise100New --pce_ckpt latest \
#         --ae_exp_name pretrained --ae_ckpt ckpt_epoch1000"


# COV - JSD
# cmd = "python pc2cad.py --exec eval --mode gen --num_worker 8  \
#         --exp_name pc2cad_contDiffNums --ckpt latest --n_points 8096 -g 0 --batch_size 32"


cmd = "python pc2cad.py --exec eval --mode acc --num_worker 8  \
        --exp_name pcEncFinalTransform_8096_500epochs --ckpt ckpt_epoch50_num8096 --n_points 8096 -g 0 "
#         --load_modular_ckpt --pce_exp_name pcEncoder_Transformation --pce_ckpt ckpt_epoch500_num8096 \
#         --ae_exp_name pretrained --ae_ckpt ckpt_epoch1000"


# cmd = "tensorboard --logdir proj_log/pc2cad/pcEncFinalTransform_8096_500epochs/log --host 0.0.0.0"

# cmd = "tensorboard --logdir proj_log/pce/pcEncoder_Transformation/log --host 0.0.0.0"

os.system(cmd)
