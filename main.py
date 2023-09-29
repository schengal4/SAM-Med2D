import streamlit as st
from streamlit_javascript import st_javascript
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates
import time

import numpy as np
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


# pip install streamlit-image-annotation (bounding box), image url option


# CODE TO LOAD SAM-2D MODELS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = Namespace()
args.device = device
args.image_size = 256
args.encoder_adapter = True
args.sam_checkpoint = "pretrain_model/sam-med2d_b.pth"

@st.cache_resource
def load_model(_args):
    """
    Loads and initializes a segmentation model based on provided arguments.
    
    This function leverages Streamlit's caching mechanism to ensure that the model is loaded only once and 
    reused across multiple calls (provided that the same arguments are again passed in), improving 
    performance and responsiveness of the app.
    
    Parameters:
    - args (object): An object containing necessary attributes to initialize the model. Expected attributes 
    include the model type and device for deployment.
    
    Returns:
    - SammedPredictor object: An instance of the SammedPredictor class initialized with the specified model.
    
    Notes:
    The function fetches the specified model from the 'sam_model_registry' and moves it to the desired device 
    (e.g., CPU or GPU). The model is set to evaluation mode using 'model.eval()' to ensure it's ready for 
    inference (it's able to segment new images based on the image points the user clicks on).
    """
    model = sam_model_registry["vit_b"](_args).to(_args.device)
    model.eval()
    predictor = SammedPredictor(model)
    return predictor

# Initializaiton code to load the models.
container_1 = st.empty()
container_1.write("Models are loading. If they aren't in the cache, they can take a few minutes to load.")
predictor_with_adapter = load_model(args) # Loads the model with the adapter layer
args.encoder_adapter = False
predictor_without_adapter = load_model(args) # Loads the similar model but without the adapter layer
container_1.write("Models are loaded.")

# Other functions used to run the app
def run_sammed(input_image, selected_points, last_mask, adapter_type):

    """
    Performs segmentation on the provided image using selected points and an optional previous mask.
    
    Parameters:
    - input_image (numpy.array): The input image on which segmentation is to be performed.
    - selected_points (list of tuples): A list of user-selected points on the image. Each tuple contains 
    the coordinates and label of a point.
    - last_mask (torch.Tensor or None): A tensor representing the previous segmentation result. If it's 
    the user's first interaction or no prior mask is available, this should be None.
    - adapter_type (str): Specifies the type of adapter to use for prediction. Determines which predictor 
    model to use (e.g., "SAM-Med2D-B").
    
    Returns:
    - list: A list containing two elements:
        1. A tuple with the original image (with segmentation mask overlaid) and a separate mask image.
        2. An updated last_mask tensor representing the latest segmentation result.
    
    Note:
    The function selects the appropriate predictor based on the adapter_type, uses it to segment the 
    input image based on the selected points and optionally the last_mask, and then visualizes the results.
    """

    if adapter_type == "SAM-Med2D-B":
        predictor = predictor_with_adapter
        # st.write("Using model with adapter layer.")
    else:
        predictor = predictor_without_adapter
        # st.write("Using model without adapter layer.")
        
    image_pil = Image.fromarray(input_image) #.convert("RGB")
    image = input_image
    # st.write(image.shape)
    try:
        H,W,_ = image.shape
    except:
        st.warning("Image is grayscale.")
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        H,W, _ = image.shape
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

    draw_points(selected_points, image_draw)

    image_pil = image_pil.convert('RGBA')
    image_pil.alpha_composite(mask_image)
    last_mask = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float, device=device))
    return [(image_pil, mask_image), last_mask]

# @st.cache_data
def draw_mask(mask, draw, random_color=False):
    """
    Draws a segmentation mask onto a given drawable surface.
    
    Parameters:
    - mask (numpy.array or similar): A mask where nonzero values indicate the segmented region.
    - draw (ImageDraw.Draw object): A drawable surface from the PIL library onto which the mask will be drawn.
    - random_color (bool, optional): If True, the mask will be drawn with a random color. If False, a default 
    blue color will be used. Defaults to False.
    
    Notes:
    The mask is visualized by coloring the pixels corresponding to nonzero values in the provided mask. The 
    transparency of the color is set to ensure some level of see-through, allowing underlying image details 
    to be visible.
    """
    if random_color:
        color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255), 153)
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)

def draw_points(points, draw, r=5):
    """
    Draws labeled points onto a given drawable surface in memory (not yet onto the Streamlit UI).
    
    Parameters:
    - points (list of tuples): A list of points where each entry is a tuple containing the coordinates and label 
    of the point. For example, [((x1, y1), label1), ((x2, y2), label2), ...].
    - draw (ImageDraw.Draw object): A drawable surface from the PIL library onto which the points will be drawn.
    - r (int, optional): The radius of the circle used to represent each point. Defaults to 5.
    
    Behavior:
    Points with a label of 1 are drawn in green, and those with a label of 0 are drawn in red. Each point is 
    visualized as an circle centered on its coordinates, with a radius specified by the 'r' parameter.
    """
    show_point = []
    for point, label in points:
        x,y = point
        if label == 1:
            draw.ellipse((x-r, y-r, x+r, y+r), fill='green')
        elif label == 0:
            draw.ellipse((x-r, y-r, x+r, y+r), fill='red')
            
def get_original_points(resized_points, scaling_factor):
    """
    Given a list of points on the resized image, get the corresponding points on the original image.

    Parameters:
    - resized_points (List of tuples in the format ((point_x, point_y), label).
        - point_x = x coordinate of point on resized image
        - point_y = y coordinate of point on resized image
        - label = 0 means background point and label = 1 means foreground point
    - scaling_factor (float): scaling_factor = resized_image_size/original_image_size
    """
    original_points = []
    try:
        for point, label in resized_points:
            original_point_x_coord, original_point_y_coord = int(point[0]/scaling_factor), int(point[1]/scaling_factor)
            original_points.append(((original_point_x_coord, original_point_y_coord), label))
    except:
        st.warning("Points don't have a corresponding foreground/background label. This may impact the program's accuracy.")
        for point in resized_points:
            original_point_x_coord, original_point_y_coord = int(point[0]/scaling_factor), int(point[1]/scaling_factor)
            original_points.append((original_point_x_coord, original_point_y_coord))
    return original_points

def attempt_rerun():
    try:
        # TODO: Fix coding style here.
        st.experimental_rerun()
    except Exception as e:
        st.error("Streamlit UI Error: Points won't display on image as expected. \
                                Please click again on the point you previously clicked on to fix this issue.")


# Initialize variables and file uploading UI
def main():
    if 'points' not in st.session_state:
        st.session_state['points'] = []
    if 'last_mask' not in st.session_state:
        st.session_state['last_mask'] = None
    if 'run_id' not in st.session_state:
        st.session_state['run_id'] = 1
    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]) 
    # Select ML model to use (with adapter or without one)
    model = st.selectbox("Select Adapter for Model", ("SAM-Med2D-B_w/o_adapter", "SAM-Med2D-B"))
    click_image = st.container()
    if uploaded_file is not None:
        with Image.open(uploaded_file) as img:
            width, height = img.size
            # Determine scaling factor to fit image appropriately to fit 
            # the screen. The except block offers 
            # a backup in case st_javascript package breaks/fails.
            # 
            
            try:
                ui_width = st_javascript("window.innerWidth")
                scaling_factor = ui_width/width
            except: 
                st.warning("Error resizing image. Using default scaling.")
                scaling_factor = 500/width
            new_width, new_height = int(width*scaling_factor), int(height*scaling_factor)

            resized_img = img.resize((new_width, new_height))
            # st.write(resized_img.size)

            a = """
            Expected behavior:

            Before: st.session_state["points"] = [((100, 101). 1)]
            After: st.session_state["points"]

            """


            # Reset all previously marked points (and masks) if user clciks on button.
            reset_points = st.button("Erase All Marked Points on the Image")
            if reset_points:
                st.session_state['points'] = []

                st.session_state['last_mask'] = None

            draw = ImageDraw.Draw(resized_img)
            
            # Draw an circle at each coordinate in points
            draw_points(st.session_state['points'], draw, r=8)


            # Select foreground or background point
            point_label = st.radio("point labels", ["Foreground Point", "Background Point"], horizontal=True)
            if point_label == "Foreground Point":
                label = 1
            elif point_label == "Background Point":
                label = 0


            # Detect where the user clicked on the image and add 
            # the respective image coordinates to st.session_state and the image itself.
            with click_image:
                # Returns the value of the previously clicked coordinates, even if on a previous run
                run_id = st.session_state['run_id'] // 2
                value = streamlit_image_coordinates(resized_img, key=str(run_id), 
                                                width = new_width, 
                                                height = new_height)
                if value is not None:
                    point = int(value["x"]), int(value["y"])
                    all_points = [pt for pt, label_ in st.session_state["points"]]
                    if (point not in all_points):
                        st.session_state["points"].append((point, label))
                        st.session_state['run_id'] += 1
                        attempt_rerun()
            # Run ML model on image with the points the user selected passed in.
            run_model = st.button("Run MedSAM-2D model.")
            if run_model:
                val1, val2 = run_sammed(np.array(img), get_original_points(st.session_state["points"], scaling_factor), 
                                        st.session_state['last_mask'], model)
                image_with_mask, mask = val1
                st.session_state['last_mask'] = val2
                st.image(image_with_mask, use_column_width=True)
            
            # st.write(st.session_state['points'])
if __name__ == "__main__": 
    main()