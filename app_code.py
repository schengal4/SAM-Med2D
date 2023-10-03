import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor
from argparse import Namespace
import torch
import torchvision
import os, sys
import random
import warnings
from scipy import ndimage
import functools
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = Namespace()
args.device = device
args.image_size = 256
args.encoder_adapter = True
args.sam_checkpoint = "pretrain_model/sam-med2d_b.pth"  #sam_vit_b.pth  sam-med2d_b.pth


def load_model(args):
    model = sam_model_registry["vit_b"](args).to(args.device)
    model.eval()
    predictor = SammedPredictor(model)
    return predictor


predictor_with_adapter = load_model(args)
args.encoder_adapter = False
predictor_without_adapter = load_model(args)

def run_sammed(input_image, selected_points, last_mask, adapter_type):
    if adapter_type == "SAM-Med2D-B":
        predictor = predictor_with_adapter
    else:
        predictor = predictor_without_adapter
        
    image_pil = Image.fromarray(input_image) #.convert("RGB")
    image = input_image
    H,W,_ = image.shape
    predictor.set_image(image)
    centers = np.array([a for a,b in selected_points ])
    point_coords = centers
    point_labels = np.array([b for a,b in selected_points ])

    masks, _, logits = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    mask_input = last_mask,
    multimask_output=True 
    ) 

    mask_image = Image.new('RGBA', (W, H), color=(0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)
    for mask in masks:
        draw_mask(mask, mask_draw, random_color=False)
    image_draw = ImageDraw.Draw(image_pil)

    draw_point(selected_points, image_draw)

    image_pil = image_pil.convert('RGBA')
    image_pil.alpha_composite(mask_image)
    last_mask = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float, device=device))
    return [(image_pil, mask_image), last_mask]

def draw_mask(mask, draw, random_color=False):
    if random_color:
        color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255), 153)
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)

def draw_point(point, draw, r=5):
    show_point = []
    for point, label in point:
        x,y = point
        if label == 1:
            draw.ellipse((x-r, y-r, x+r, y+r), fill='green')
        elif label == 0:
            draw.ellipse((x-r, y-r, x+r, y+r), fill='red')

colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]
block = gr.Blocks()
with block:
    with gr.Row():
        gr.Markdown(
            '''# SAM-Med2D!🚀
            SAM-Med2D is an interactive segmentation model based on the SAM model for medical scenarios, supporting multi-point interactive segmentation and box interaction. 
            Currently, only multi-point interaction is supported in this application. More information can be found on [**GitHub**](https://github.com/uni-medical/SAM-Med2D/tree/main).
            '''
        )
        with gr.Row():
            # select model
            adapter_type = gr.Dropdown(["SAM-Med2D-B", "SAM-Med2D-B_w/o_adapter"], value='SAM-Med2D-B', label="Select Adapter")
            # adapter_type.change(fn = update_model, inputs=[adapter_type])
          
    with gr.Tab(label='Image'):
        with gr.Row().style(equal_height=True):
            with gr.Column():
                # input image
                original_image = gr.State(value=None)   # store original image without points, default None
                input_image = gr.Image(type="numpy")
                # point prompt
                with gr.Column():
                    selected_points = gr.State([])      # store points
                    last_mask = gr.State(None) 
                    with gr.Row():
                        gr.Markdown('You can click on the image to select points prompt. Default: foreground_point.')
                        undo_button = gr.Button('Undo point')
                    radio = gr.Radio(['foreground_point', 'background_point'], label='point labels')
                button = gr.Button("Run!")
        
            gallery_sammed = gr.Gallery(
                    label="Generated images", show_label=False, elem_id="gallery").style(preview=True, grid=2,object_fit="scale-down")
            
    def process_example(img):
        return img, [], None    
    
    def store_img(img):
        return img, [], None  # when new image is uploaded, `selected_points` should be empty
    input_image.upload(
        store_img,
        [input_image],
        [original_image, selected_points, last_mask]
    )
    # user click the image to get points, and show the points on the image
    def get_point(img, sel_pix, point_type, evt: gr.SelectData):
        if point_type == 'foreground_point':
            sel_pix.append((evt.index, 1))   # append the foreground_point
        elif point_type == 'background_point':
            sel_pix.append((evt.index, 0))    # append the background_point
        else:
            sel_pix.append((evt.index, 1))    # default foreground_point
        # draw points
        for point, label in sel_pix:
            cv2.drawMarker(img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
        # if img[..., 0][0, 0] == img[..., 2][0, 0]:  # BGR to RGB
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img if isinstance(img, np.ndarray) else np.array(img)
    
    input_image.select(
        get_point,
        [input_image, selected_points, radio],
        [input_image],
    )

    # undo the selected point
    def undo_points(orig_img, sel_pix):
        if isinstance(orig_img, int):   # if orig_img is int, the image if select from examples
            temp = cv2.imread(image_examples[orig_img][0])
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        else:
            temp = orig_img.copy()
        # draw points
        if len(sel_pix) != 0:
            sel_pix.pop()
            for point, label in sel_pix:
                cv2.drawMarker(temp, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
        if temp[..., 0][0, 0] == temp[..., 2][0, 0]:  # BGR to RGB
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        return temp, None if isinstance(temp, np.ndarray) else np.array(temp), None
    undo_button.click(
        undo_points,
        [original_image, selected_points],
        [input_image, last_mask]
    )

    with gr.Row():
        with gr.Column():
            gr.Examples(["data_demo/images/amos_0507_31.png", "data_demo/images/s0114_111.png" ], inputs=[input_image], outputs=[original_image, selected_points,last_mask], fn=process_example, run_on_click=True)

    button.click(fn=run_sammed, inputs=[original_image, selected_points, last_mask, adapter_type], outputs=[gallery_sammed, last_mask])

block.launch(debug=True, share=True, show_error=True)

