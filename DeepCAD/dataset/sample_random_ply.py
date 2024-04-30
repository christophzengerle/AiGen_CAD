import json
import os
import random
import shutil

path = "data/train_val_test_split.json"
phase = "validation"
with open(path, "r") as fp:
    all_data = json.load(fp)[phase]

sample = random.sample(all_data, 10)

# sample = ["0000/00000007", "0000/00001271", "0000/00001506", "0000/00001817"]

for file in sample:
    pc_path = os.path.join("data/cad_pc", file + ".ply")
    shutil.copyfile(
        pc_path, os.path.join("evaluation/eval_pc_cad_images/source", pc_path.split("/")[-1])
    )
