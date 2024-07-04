# AiGen-CAD - Generative AI for 3d modeling
Project structure:
```
AiGen_CAD
│   docker-compose.yaml
│ 
│   
└───App (Gradio UI) 
│       │   Dockerfile
│       │   requirements.txt
│       │   ...
│   
└───DeepCAD
│       │   Dockerfile
│       │   requirements.txt
│       │   ...
│   
└───InstantMesh
│       │   Dockerfile
│       │   requirements.txt
│       │   ...
```

Every subdirectory has its own Dockerfile and requirements.txt to initiate Docker Container. 
Every subdirectory is mounted to their respective container when building the container with Docker Compose. 

## Docker
Running 
```bash
$ docker compose up
```
in AiGen_CAD directory starts all 3 container for InstantMesh, DeepCAD and the GradioUI.
The access point scripts of InstantMesh and DeepCAD will be executed automatically. 
To access the Container bash run "docker exec -it {container_name} bash".   

The Containernames are:
* app
* instantmesh
* deepcad  

The structure of the containers is like: 

```
usr
│
└───/local/cuda (only for DeepCAD. CudaHome directory)
│
└───/app/src
    │
    └───miniconda3
    │   
    └───InstantMesh/App/DeepCAD (mounte from respective subdirectory)
        │
        └───data (mounted from ../utils/data)
        │
        └───ckpts/proj_log (mounted from ../utils/models/{InstantMesh/DeepCAD})
        │
        └───results (mounted from ../utils/results
```

The docker-compose.yaml manages the ports for every container. The App container has SSH and Gradio Ports.
The InstantMesh and DeepCAD container have SSH, Flask and Tensorboard ports each. 


## Data Transformation

### Command Sequence (.json/.h5) to CAD (.step)
file: DeepCAD/utils/seq2step.py  
Takes Commands Sequences as input and transforms them to Mesh in OBJ-format.

Parameters:  
* **src : str, default=None**  
Source file or folder (takes every .json/.h5 file in the directory as input)
* **dest: str, default="step_files"**  
Destination folder. Is created if doesn't exist.  
* **type: str, default=h5, choices=[h5, json]**  
Select file format of input (json or h5) 
* **check: bool, default=None**  
Use opencascade analyzer to filter invalid model
select file format of input (json or h5) 


### CAD (.step) to Mesh (.obj)
file: InstantMesh/src/utils/step2obj.py  
Takes CAD Step-Files as input and transforms it to Mesh in OBJ-format. 

Parameters:  
* **src : str, default=None**  
Source file or folder (takes every .step file in the directory as input)
* **dest: str, default="png_files"**  
Destination folder. Is created if doesn't exist.


### CAD or Mesh (.step/.obj) to Pointcloud (.ply)
file: InstantMesh/src/utils/step2pc.py  
Takes CAD Step-Files or Mesh OBJ-Files and transforms them to Pointclouds in PLY-format.

Parameters:  
* **src : str, default=None**  
Source file or folder (takes every .step/.obj file in the directory as input)
* **dest: str, default="ply_files"**  
Destination folder. Is created if doesn't exist.


### CAD, Mesh or Pointcloud (.step/.obj/.ply) to Image/Video (.png/.gif)
file: DeepCAD/utils/step2render.py  
Takes CAD Step-Files, Mesh OBJ-Files or Pointcloud PLY-Files and transforms them to rendered Image or Video. Can also save input as OBJ-File. 

Parameters:  
* **src : str, default=None**  
Source file or folder (takes every .step/.obj/.ply file in the directory as input)
* **dest: str, default="png_files"**  
Destination folder. Is created if doesn't exist.
* **ele: int, default=45**  
Camera elevation.
* **rot: int, default=-45**  
Camera rotation.
* **png: bool, default=False**  
If True renders and saves PNG-File.
* **gif: bool, default=False**  
If True renders and saves GIF-File.
* **obj: bool, default=False**  
If True saves Mesh as OBJ-File.
* **qual: str, default="low", choices=["low", "medium", "high"]**  
Quality of render. Low is 300, medium 600 and high 1200 pixel. 


### Mesh (.obj) to Edge-Image, Depth-Image and Normal-Image (.png)
file: InstantMesh/src/utils/mesh2instant.py  
Takes Mesh OBJ-File and renders Edge-, Depth- and Normal-Image. 
Takes train-test-split Json-File as input and saves filenames of split in 
*val_objs.json* under keywords *good_objs*, *val_objs*, *test_objs* and *failed_objs* for files where render failes. 

Parameters:  
* **src : str, default=None**  
Source file or folder (takes every .step/.obj file in the directory as input)
* **dest: str, default="png_files"**  
Destination folder. Is created if doesn't exist.
* **res: str, default="low", choices=["low", "medium", "high"]**  
Quality of render. Low is 300, medium 600 and high 1200 pixel. 
* **split: str, is required**  
Train-test-split file. Structure like dict: {"train": [{folder}/{file}, {folder}/{file},...], "val": [...], "test": [...]}  
Filename without file ending! Folder path in relation to src folder!

## App
To start the Pipeline you first need to wait until the access points of InstantMesh and DeepCAD are loaded. 
If no cached models are available downloading and initializing the models can take a few minutes. 
Cashed models will me saved in the *proj_log* folder for DeepCAD and the *ckpts* folder for InstantMesh. 
These folders are mounted so the cached models should be available even after container restart. 
To run the Gradio UI log into the App container bash as descriped in **Docker** and run 
```bash
$ python app.py
```
The terminal should generate a URL.


## DeepCAD

## InstantMesh

### **Run InstantMesh Gradio App**:  
file: InstantMesh/app.py

```bash
$ python app.py
```

### **Run Training**:  
file: InstantMesh/train.py

Command to run basic training:
```bash
$ python train.py --base configs/instant-mesh-base-train.yaml --gpus 0, 
```

The checkpoint path, the data source and the pytorch-lightning settings are given in the config file. 
  
Parameters: 
* **resume: str, default=None**  
Resume from checkpoint. Only for nerf model.
* **resume_weights_only: str**  
Only resume model weights. Only for nerf model. 
* **base: str, default="base_config.yaml"**  
Path to base configs. 
* **name: str, default: ""**  
Experiment name.
* **num_nodes: int, default=1**  
Numbers of nodes to use.
* **gpus: str, default="0,"**  
GPU ids to use. On IDF cluster only works with one GPU!
* **seed: str, default=42**  
Seed for seed_everything. 
* **logdir: str, default: "logs"**  
Directory for logging data.


