import os

import gradio as gr
import requests

import datetime

import base64
from PIL import Image
import io


# URLs der Backend-Services
INSTANT_MESH_URL = 'http://instantmesh:5001/'
DEEP_CAD_URL = 'http://deepcad:5002/'


#TODO
OUTPUT_FOLDER = "data/pipeline_results/"

# def instant_dummy(input_images, out_path):
#     return "data/obj_files/00000007.obj", "data/obj_files/00000007.obj"
#
#
# def deep_dummy(pc, out_path):
#     return "data/obj_files/00000007.obj", "data/obj_files/00000007.obj"


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")


def setup_dir(custom_path, save_only_last):
    folder = "aigen_cad_output" if save_only_last else str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    output_path = os.path.join(OUTPUT_FOLDER, folder) if not custom_path else os.path.join(custom_path, folder)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path

    
def preprocess(input_image, do_remove_background):
    buffered = io.BytesIO()
    input_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    data = {
        'image' : img_str,
        'do_remove_background' : do_remove_background        
    }
    response = requests.post(INSTANT_MESH_URL + "/Preprocessing", json=data)
    response_data = response.json()
    
    processed_image_data = base64.b64decode(response_data['image'])
    processed_image = Image.open(io.BytesIO(processed_image_data))
    
    return processed_image
    

def generate_mvs(input_image, sample_steps, sample_seed, out_path):
    buffered = io.BytesIO()
    input_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    data = {
        'image' : img_str,
        'sample_steps' : sample_steps,
        'sample_seed' : sample_seed       
    }
    response = requests.post(INSTANT_MESH_URL + "/GenerateMultiViews", json=data)
    response_data = response.json()

    processed_z123_image = Image.open(io.BytesIO(base64.b64decode(response_data['z123_image'])))
    processed_show_image = Image.open(io.BytesIO(base64.b64decode(response_data['show_image'])))

    processed_z123_image.save(os.path.join(out_path, "generated_multiview_show.png"))
    
    return processed_z123_image, processed_show_image


def make3d(images, out_path):
    images_data = []
    for image in images:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        images_data.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
        
    data = {
        'images' : images_data,
        'out_path' : out_path     
    }
    response = requests.post(INSTANT_MESH_URL + "/Generate3D", json=data)
    response_data = response.json()
    mesh_fpath = response_data['mesh_fpath']
    return mesh_fpath


def obj2pc(obj_path, out_path):
    data = {
        'obj_path' : obj_path,
        'out_path' : out_path     
    }
    response = requests.post(INSTANT_MESH_URL + "/Object2PointCloud", json=data)
    response_data = response.json()
    path = response_data['path']
    return path, path


def generate_cad(pc_path, out_path):
    data = {
        'pc_path' : pc_path,
        'output_path' : out_path
    }
    response = requests.post(DEEP_CAD_URL + "/GenerateCAD", json=data)
    response_data = response.json()
    STEP_path = response_data['STEP_path']
    
    data = {
        'step_path' : STEP_path,
        'out_path' : "deepCAD"     
    }
    response = requests.post(INSTANT_MESH_URL + "/STEP2Object", json=data)
    response_data = response.json()
    obj_path = response_data['obj_path']
    return obj_path, STEP_path


def provide_files(*args):
    file_list = [file for file in args]
    return file_list


#############################################################################################
###################               APPLICATION             ##################################
############################################################################################


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
                    save_only_last = gr.Checkbox(
                        label="save only last output", value=False
                    )
                    custom_path = gr.Textbox(
                        label="save path"
                    )
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
                        os.path.join("examples", img_name) for img_name in sorted(os.listdir("examples"))
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
                output_gradio_obj = gr.Model3D(
                    label="InstantMesh output",
                    # width=768,
                    interactive=False
                )

            with gr.Row():
                output_cad = gr.Model3D(
                    label="DeepCAD output",
                    interactive=False
                )

            with gr.Row():
                output_files = gr.Files(
                    label="Generated Files",
                    interactive=False
                )


    gr.Markdown(_CITE_)
    mv_images = gr.State()
    point_cloud = gr.State()
    output_path = gr.State()
    output_step = gr.State()
    output_model_obj = gr.State()

    submit.click(
        fn=check_input_image,
        inputs=[input_image]
    ).success(
        fn=setup_dir,
        inputs=[custom_path, save_only_last],
        outputs=[output_path]
    ).success(
        fn=preprocess,
        inputs=[input_image, do_remove_background],
        outputs=[processed_image],
    ).success(
        fn=generate_mvs,
        inputs=[processed_image, sample_steps, sample_seed, output_path],
        outputs=[mv_images, mv_show_images],
    ).success(
        fn=make3d,
        inputs=[mv_images, output_path],
        outputs=[output_gradio_obj, output_model_obj]
    ).success(
        fn=obj2pc,
        inputs=[output_model_obj, output_path],
        outputs=[point_cloud]
    ).success(
        fn=generate_cad,
        inputs=[point_cloud, output_path],
        outputs=[output_cad, output_step]
    ).success(
        fn=provide_files,
        inputs=[output_model_obj, point_cloud, output_step],
        outputs=[output_files]
    )

demo.queue(max_size=10)
# demo.launch(server_name="0.0.0.0", server_port=7860)
demo.launch(share=True)
