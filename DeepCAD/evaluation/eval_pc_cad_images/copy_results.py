import os
import shutil

path = "../../proj_log/pc2cad/pc2cad_Exp/results/pc2cad/latest/source_2024-05-07-10-16-10"

dest_path = "results/"
model_desc = "pc2cad/pc2cad_Exp"

dest = os.path.normpath(os.path.join(dest_path, model_desc))
if not os.path.isdir(dest):
    os.mkdir(dest)
for file in os.listdir(path):
    if file.endswith(".png") or file.endswith(".gif"):
        shutil.copyfile(os.path.join(path, file), os.path.join(dest, file))