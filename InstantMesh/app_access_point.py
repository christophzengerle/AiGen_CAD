import os
import imageio
import numpy as np
import torch
import rembg
import datetime
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics,
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_glb
from src.utils.infer_util import remove_background, resize_foreground, images_to_video
from src.utils import step2obj

from huggingface_hub import hf_hub_download
import sysconfig
import trimesh


import base64
import io
from flask import Flask, request, make_response, jsonify

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def init():
    return make_response("InstantMesh running...", 200)

###############################################################################
# Configuration.
###############################################################################

print(sysconfig.get_paths()['include'])

if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')
else:
    device0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device1 = device0

# Define the cache directory for model files
instantmesh_cache_dir = './ckpts/'
os.makedirs(instantmesh_cache_dir, exist_ok=True)


seed_everything(0)

config_path = './configs/instant-mesh-large.yaml'
config = OmegaConf.load(config_path)
config_name = os.path.basename(config_path).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

device = torch.device('cuda')


# load diffusion model
print('Loading diffusion model ...')
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline="./zero123plus",
    torch_dtype=torch.float16,
    cache_dir=instantmesh_cache_dir
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

# load custom white-background UNet
unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin",
                                repo_type="model", cache_dir=instantmesh_cache_dir)
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)

pipeline = pipeline.to(device0)

# load reconstruction model
print('Loading reconstruction model ...')
model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_large.ckpt",
                                repo_type="model", cache_dir=instantmesh_cache_dir)
model = instantiate_from_config(model_config)
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
model.load_state_dict(state_dict, strict=True)

model = model.to(device1)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device1, fovy=30.0)
model = model.eval()



@app.route('/Preprocessing', methods=['POST'])
def preprocess():
    #TODO: Preprocessed Bilder abspeichern?
    # input_image, do_remove_background
    data = request.json
    
    image = base64.b64decode(data['image'])
    input_image = Image.open(io.BytesIO(image))
    
    do_remove_background = data['do_remove_background']    

    rembg_session = rembg.new_session() if do_remove_background else None
    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)
        
    encoded_image = base64.b64decode(input_image).decode('utf-8')
    
    response = {
        'image' : encoded_image
    }
    
    return jsonify(response)


@app.route('/GenerateMultiViews', methods=['POST'])
def generate_mvs():
    # input_image, sample_steps, sample_seed
    data = request.json
    
    image = base64.b64decode(data['image'])
    input_image = Image.open(io.BytesIO(image))
    
    sample_steps = data['sample_steps'] 
    sample_seed = data['sample_seed'] 
    
    
    seed_everything(sample_seed)

    # sampling
    generator = torch.Generator(device=device0)
    z123_image = pipeline(
        input_image,
        num_inference_steps=sample_steps,
        generator=generator,
    ).images[0]

    show_image = np.asarray(z123_image, dtype=np.uint8)
    show_image = torch.from_numpy(show_image)  # (960, 640, 3)
    show_image = rearrange(show_image, '(n h) (m w) c -> (n m) h w c', n=3, m=2)
    show_image = rearrange(show_image, '(n m) h w c -> (n h) (m w) c', n=2, m=3)
    show_image = Image.fromarray(show_image.numpy())
    
    processed_z123_images_data = []
    processed_show_images_data = []
    
    for image in z123_image:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        processed_z123_images_data.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
        
    for image in show_image:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        processed_show_images_data.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
    
    response = {
        'z123_image' : processed_z123_images_data,
        'show_image' : processed_show_images_data
    }
    
    return jsonify(response)



@app.route('/Generate3D', methods=['POST'])
def make3d():
    # images, out_path
    data = request.json
    
    images_data = data['images']
    out_path = data['out_path']
    
    images = [Image.open(io.BytesIO(base64.b64decode(img))) for img in images_data]    
    
    
    images = np.asarray(images, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()  # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)  # (6, 3, 320, 320)

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device1)
    render_cameras = get_render_cameras(
        batch_size=1, radius=4.5, elevation=20.0, is_flexicubes=IS_FLEXICUBES).to(device1)

    images = images.unsqueeze(0).to(device1)
    images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)

    mesh_fpath = os.path.join(out_path, "instantMesh.obj")
    print(mesh_fpath)
    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    video_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.mp4")

    with torch.no_grad():
        # get triplane
        planes = model.forward_planes(images, input_cameras)

        # get video
        chunk_size = 20 if IS_FLEXICUBES else 1
        render_size = 384

        frames = []
        for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
            if IS_FLEXICUBES:
                frame = model.forward_geometry(
                    planes,
                    render_cameras[:, i:i + chunk_size],
                    render_size=render_size,
                )['img']
            else:
                frame = model.synthesizer(
                    planes,
                    cameras=render_cameras[:, i:i + chunk_size],
                    render_size=render_size,
                )['images_rgb']
            frames.append(frame)
        frames = torch.cat(frames, dim=1)

        images_to_video(
            frames[0],
            video_fpath,
            fps=30,
        )

        print(f"Video saved to {video_fpath}")

    mesh_fpath, mesh_glb_fpath = make_mesh(mesh_fpath, planes)

    response = {
        'mesh_fpath' : mesh_fpath
    }
    
    return jsonify(response)


@app.route('/Object2PointCloud', methods=['POST'])
def obj2pc():
    # obj_path, out_path
    data = request.json
    obj_path = data['obj_path']
    out_path = data['out_path']
    m = trimesh.load_mesh(obj_path)
    path = os.path.join(out_path, "instantMesh.ply")
    m.export(path, file_type='ply')
    response = {
        'path' : path
    }
    
    return jsonify(response)


@app.route('/STEP2Object', methods=['POST'])
def step2Obj():
    # obj_path, out_path
    data = request.json
    step_path = data['step_path']
    out_path = data['out_path']
    obj_path = step2obj.transform(step_path, "deepCAD")
    response = {
        'obj_path' : obj_path
    }
    
    return jsonify(response)






def get_render_cameras(batch_size=1, M=120, radius=2.5, elevation=10.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def make_mesh(mesh_fpath, planes):
    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    mesh_glb_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.glb")

    with torch.no_grad():
        # get mesh

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=False,
            **infer_config,
        )

        vertices, faces, vertex_colors = mesh_out
        vertices = vertices[:, [1, 2, 0]]

        save_glb(vertices, faces, vertex_colors, mesh_glb_fpath)
        save_obj(vertices, faces, vertex_colors, mesh_fpath)

        print(f"Mesh saved to {mesh_fpath}")

    return mesh_fpath, mesh_glb_fpath



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
