import os
import imageio
import numpy as np
import torch
import rembg
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from InstantMesh.src.utils.train_util import instantiate_from_config
from InstantMesh.src.utils.camera_util import (
    FOV_to_intrinsics,
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from InstantMesh.src.utils.mesh_util import save_obj, save_glb
from InstantMesh.src.utils.infer_util import remove_background, resize_foreground, images_to_video

import tempfile
from huggingface_hub import hf_hub_download
import sysconfig
import trimesh

print(sysconfig.get_paths()['include'])

if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')
else:
    device0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device1 = device0

# Define the cache directory for model files
model_cache_dir = './ckpts/'
os.makedirs(model_cache_dir, exist_ok=True)


def instant_dummy(images):
    path = "data/obj_files/0000/00000007.obj"
    return path


def obj2pc(obj_path):
    m = trimesh.load_mesh(obj_path)
    path = "data/temp.ply"
    m.export("data/temp.ply")
    return path


def generate_deepcad(pc_path):
    return "data/obj_files/0000/00000069.obj"

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


def images_to_video(images, output_path, fps=30):
    # images: (N, C, H, W)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for i in range(images.shape[0]):
        frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
        assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
            f"Frame shape mismatch: {frame.shape} vs {images.shape}"
        assert frame.min() >= 0 and frame.max() <= 255, \
            f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        frames.append(frame)
    imageio.mimwrite(output_path, np.stack(frames), fps=fps, codec='h264')


###############################################################################
# Configuration.
###############################################################################

seed_everything(0)

config_path = 'InstantMesh/configs/instant-mesh-large.yaml'
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
    custom_pipeline=os.path.join("InstantMesh", "zero123plus"),
    torch_dtype=torch.float16,
    cache_dir=model_cache_dir
)
print("test")
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)

# load custom white-background UNet
unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin",
                                 repo_type="model", cache_dir=model_cache_dir)
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)

pipeline = pipeline.to(device0)

# load reconstruction model
print('Loading reconstruction model ...')
model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_large.ckpt",
                                  repo_type="model", cache_dir=model_cache_dir)
model = instantiate_from_config(model_config)
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
model.load_state_dict(state_dict, strict=True)

model = model.to(device1)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device1, fovy=30.0)
model = model.eval()

print('Loading Finished!')


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def preprocess(input_image, do_remove_background):
    rembg_session = rembg.new_session() if do_remove_background else None
    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)

    return input_image


def generate_mvs(input_image, sample_steps, sample_seed):
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

    return z123_image, show_image


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


def make3d(images):
    images = np.asarray(images, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()  # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)  # (6, 3, 320, 320)

    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device1)
    render_cameras = get_render_cameras(
        batch_size=1, radius=4.5, elevation=20.0, is_flexicubes=IS_FLEXICUBES).to(device1)

    images = images.unsqueeze(0).to(device1)
    images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)

    mesh_fpath = tempfile.NamedTemporaryFile(suffix=f".obj", delete=False).name
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

    return video_fpath, mesh_fpath, mesh_glb_fpath


import gradio as gr

_HEADER_ = '''
<h2><b>AiGen-CAD Demo</b></h2>
'''

_CITE_ = r"""
"""

with gr.Blocks() as demo:
    gr.Markdown(_HEADER_)
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Input Image",
                    image_mode="RGBA",
                    sources="upload",
                    width=256,
                    height=256,
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(
                    label="Processed Image",
                    image_mode="RGBA",
                    width=256,
                    height=256,
                    type="pil",
                    interactive=False
                )
            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    sample_seed = gr.Number(value=42, label="Seed Value", precision=0)

                    sample_steps = gr.Slider(
                        label="Sample Steps",
                        minimum=30,
                        maximum=75,
                        value=75,
                        step=5
                    )

            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")

            with gr.Row(variant="panel"):
                gr.Examples(
                    examples=[
                        os.path.join("InstantMesh", "examples", img_name) for img_name in sorted(os.listdir(os.path.join("InstantMesh", "examples")))
                    ],
                    inputs=[input_image],
                    label="Examples",
                    examples_per_page=20
                )

        with gr.Column():
            with gr.Row():
                with gr.Column():
                    mv_show_images = gr.Image(
                        label="Generated Multi-views",
                        type="pil",
                        # width=379,
                        interactive=False
                    )

                # with gr.Column():
                #     output_video = gr.Video(
                #         label="video", format="mp4",
                #         width=379,
                #         autoplay=True,
                #         interactive=False
                #     )

            with gr.Row():
                output_model_obj = gr.Model3D(
                    label="InstantMesh output",
                    # width=768,
                    interactive=False
                )

            with gr.Row():
                output_cad = gr.Model3D(
                    label="DeepCAD output",
                    interactive=False
                )

    gr.Markdown(_CITE_)
    mv_images = gr.State()
    point_cloud = gr.State()

    submit.click(fn=check_input_image, inputs=[input_image]).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background],
        outputs=[processed_image],
    ).success(
        fn=generate_mvs,
        inputs=[processed_image, sample_steps, sample_seed],
        outputs=[mv_images, mv_show_images],
    ).success(
        fn=instant_dummy,
        inputs=[mv_images],
        outputs=[output_model_obj]
    ).success(
        fn=obj2pc,
        inputs=[output_model_obj],
        outputs=[point_cloud]
    ).success(
        fn=generate_deepcad,
        inputs=[point_cloud],
        outputs=[output_cad]
    )

demo.queue(max_size=10)
# demo.launch(server_name="0.0.0.0", server_port=7860)
demo.launch()
