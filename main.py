import streamlit as st
import io
container_1 = st.empty()
container_1.info("Models are loading. If they aren't in the cache, they can take a few minutes to load.")


from streamlit_javascript import st_javascript
import urllib.request
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
# from huggingface_hub import hf_hub_download
# hf_hub_download("schengal1/SAM-Med2D_model", "sam-med2d_b.pth")
# print(os.getcwd())
# time.sleep(200)



# Couple constants for readability purposes.
FOREGROUND_POINT = 1
BACKGROUND_POINT = 0

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
predictor_with_adapter = load_model(args) # Loads the model with the adapter layer
args.encoder_adapter = False
predictor_without_adapter = load_model(args) # Loads the similar model but without the adapter layer
container_1.info("Models are loaded.") # See statement below import streamlit as st for where container_1 is declared.

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
# TODO: Draw label = 0 points as red crosses instead of red circles. Also, eliminate hardcoding
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
        if label == FOREGROUND_POINT:
            draw.ellipse((x-r, y-r, x+r, y+r), fill='green')
        elif label == BACKGROUND_POINT:
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
    """
    Tries to stop the execution of streamlit script and then rerun from the beginning.
    """
    try:
        st.experimental_rerun()
    except Exception as e:
        st.error("Streamlit UI Error: Points won't display on image as expected. \
                                Please click again on the point you previously clicked on to fix this issue.")
def reset_points_and_masks():
    st.session_state['points'] = []
    st.session_state['last_mask'] = None
def initialize_styling():
    # TODO: image-with-the-indicated region segmented is the id of the subheader. if the subheader title changes, the id will change and some of the CSS below won't work.
    try:
        st.markdown("""
        <style>
        .stRadio [role=radiogroup] {
            align-items: center;
            justify-content: center;
        }
        .stRadio label {
            align-items: center;
            justify-content: center;
        }
        .stButton button {
            display: block;
            margin-left: auto;
            margin-right: auto;
            align-items: center;
        }
        #image-with-the-indicated-region-segmented {
            text-align: center;        
        }
        </style>
    """,unsafe_allow_html=True)
    except:
        st.warning("Centering of radio buttons isn't working. The app will still work, though the layout might be slightly off.")
# Initialize variables and file uploading UI
def main():
    initialize_styling()
    if 'points' not in st.session_state:
        st.session_state['points'] = []
    if 'last_mask' not in st.session_state:
        st.session_state['last_mask'] = None
    if 'run_id' not in st.session_state:
        st.session_state['run_id'] = 1
    if 'uploaded_file_id' not in st.session_state:
        st.session_state['uploaded_file_id'] = None
    if 'uploaded_image_URL' not in st.session_state:
        st.session_state['uploaded_image_URL'] = None
    # File uploader widget or get image from URL
    input_choice = st.radio("Do you want to ...", ["Upload an image (.jpg, .jpeg, or .png)?", "Type in the image URL?"], horizontal=True)
    if ("Upload an image" in input_choice):
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        uploading_file_progress_message = st.empty()
        container_1.empty()
        if uploaded_file is not None and uploaded_file.file_id != st.session_state['uploaded_file_id']:
            uploading_file_progress_message.info("Loading image and preparing to display it. This can take several seconds.")
            reset_points_and_masks()
            st.session_state['uploaded_file_id'] = uploaded_file.file_id
    elif input_choice == "Type in the image URL?":
            image_url = st.text_input("Enter the URL of the image you want to upload below (e.g., https://pressbooks.pub/app/uploads/sites/3987/2017/10/chest-case-12-2-909x1024.jpg)", help = "To get the URL of a Google Search image, right-click on it. Then, select/click on 'Copy Image Address'.")
            uploading_file_progress_message = st.empty()
            container_1.empty()
            if image_url.strip() != "": # Checks if the user entered some text for the image URL
                try:
                    urllib.request.urlretrieve(image_url, "__image.png")
                except: 
                    st.error("The URL you entered isn't working.") # If the user enters an invalid URL
                    uploaded_file = None
                else:
                    uploaded_file = "__image.png"
                    if image_url != st.session_state['uploaded_image_URL']:
                        uploading_file_progress_message.info("Loading image and preparing to display it. This can take several seconds.")
                        st.session_state['uploaded_image_URL'] = image_url
                        reset_points_and_masks()
            else:
                uploaded_file = None
            
    # Select ML model to use (with adapter or without one)
    model = st.selectbox("Select Adapter for Model", ("SAM-Med2D-B_w/o_adapter", "SAM-Med2D-B"))
    click_image = st.container()
    if uploaded_file is not None:
        with Image.open(uploaded_file) as img:
            width, height = img.size
            # Determine scaling factor to fit image appropriately to fit the screen. The except block offers
            # a backup in case st_javascript package breaks/fails.
            try:
                ui_width = st_javascript("window.innerWidth")
                scaling_factor = ui_width/width
            except: 
                st.warning("Error resizing image. Using default scaling.")
                scaling_factor = 500/width
            new_width, new_height = int(width*scaling_factor), int(height*scaling_factor)
            resized_img = img.resize((new_width, new_height))

            # Select foreground or background point. Foreground point 
            # means a point in the region the user wants to segment. 
            # Background points are points that shouldn't be in the region 
            # the user wants to segment. 
            point_label = st.radio("**Point Labels**", ["Foreground Point", "Background Point"], horizontal=True)
            if point_label == "Foreground Point":
                label = FOREGROUND_POINT
            elif point_label == "Background Point":
                label = BACKGROUND_POINT

            # Reset all previously marked points (and masks) if user clicks on button.
            left_btn, right_btn = st.columns(2)
            reset_points = left_btn.button("Erase All Marked Points on the Image")
            if reset_points:
                reset_points_and_masks()
            draw = ImageDraw.Draw(resized_img)
            
            # Draw an circle at each coordinate in points
            draw_points(st.session_state['points'], draw, r=8)

            # Detect where the user clicked on the image and add 
            # the respective image coordinates to st.session_state and the image itself.
            with click_image:
                # streamlit_image_coordinates returns the value of the previously clicked coordinates, even if on a 
                # previous run. This was causing some unexpected UI behavior. So, the run_id = st.session_state['run_id'] 
                # statement and the later statement incrementing st.session_state['run_id'] += 1 are meant to correct this.
                run_id = st.session_state['run_id']
                value = streamlit_image_coordinates(resized_img, key=str(run_id), 
                                                width = new_width, 
                                                height = new_height)
                uploading_file_progress_message.empty()
                if value is not None:
                    point = int(value["x"]), int(value["y"])
                    all_points = [pt for pt, label_ in st.session_state["points"]]
                    if (point, label) not in st.session_state['points']:
                        st.session_state["points"].append((point, label))
                        st.session_state['run_id'] += 1
                        attempt_rerun()
            # Run ML model on image with the points the user selected passed in.
            run_model = right_btn.button("Run SAM-Med2D model.")
            if run_model:
                running_model = st.empty()
                running_model.info("Running the image segmentation algorithm. This can take several seconds.")
                all_labels = [label_ for pt, label_ in st.session_state["points"]]
                if len(st.session_state["points"]) == 0:
                     st.error("Please select at least one foreground point in the above image before running the model.\
                             To do this, click the 'Foreground Point' button and then click a point on the image.")
                elif FOREGROUND_POINT not in all_labels:
                    st.error("Please select at least one foreground point in the above image before running the model.\
                             To do this, click the 'Foreground Point' button and then click a point on the image.")
                else:
                    val1, val2 = run_sammed(np.array(img), get_original_points(st.session_state["points"], scaling_factor), # img = original image that wasn't modified or drawn upon
                                        st.session_state['last_mask'], model)
                    image_with_mask, mask = val1
                    st.session_state['last_mask'] = val2
                    st.divider()
                    st.subheader("Image with the Indicated Region Segmented.")
                    st.image(image_with_mask, use_column_width=True)

                    buf = io.BytesIO()
                    image_with_mask.save(buf, format="PNG")
                    image_bytes = buf.getvalue()
                    btn = st.download_button(
                        label="Download image",
                        data=image_bytes,
                        mime="image/png"
                    )
                running_model.empty()
                    
if __name__ == "__main__": 
    main()