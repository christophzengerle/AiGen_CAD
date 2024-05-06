import os
import shutil

path = "./proj_log/pce/pcEncoder/results/pcEncodings/latest/source_2024-05-02-19-47-33"
dest = "./evaluation/eval_pc_cad_images/results_source"
for file in os.listdir(path):
    if file.endswith(".png"):
        shutil.copyfile(os.path.join(path, file), os.path.join(dest, file))