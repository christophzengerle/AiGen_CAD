import glob
import json
import os
import random
import shutil

source_path = "./evaluation/eval_testimages/source/"
files = glob.glob(source_path + "*")
for f in files:
    os.remove(f)

path = "./data/train_val_test_split.json"
phase = "validation"
with open(path, "r") as fp:
    all_data = json.load(fp)[phase]

sample = random.sample(all_data, 100)

# sample = ["0000/00000007", "0000/00001271", "0000/00001506", "0000/00001817"]

for file in sample:
    pc_path = os.path.join("./data/cad_pc", file + ".ply")
    shutil.copyfile(pc_path, os.path.join(source_path, pc_path.split("/")[-1]))
