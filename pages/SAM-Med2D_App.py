# Streamlit related imports
import streamlit as st

container_1 = st.empty()
container_1.info("Models are loading. If they aren't in the cache, they can take several minutes to load.")

from streamlit_javascript import st_javascript
from streamlit_drawable_canvas import st_canvas

# Image processing and IO related imports
import io
from PIL import Image, ImageDraw
import urllib.request
import requests
from scipy import ndimage

# Data manipulation and utility imports
import numpy as np
import pandas as pd
import os
import time
import random
import threading

# Deep Learning and model related imports
from segment_anything import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor
from argparse import Namespace
import torch

# Custom module imports
from Instructions import write_introduction




# Couple constants for readability purposes.
FOREGROUND_POINT = 1
BACKGROUND_POINT = 0

# CODE TO LOAD SAM-2D MODELS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = Namespace()
args.device = device
args.image_size = 256
args.encoder_adapter = True
args.sam_checkpoint = "sam-med2d_b.pth"
sources = """
A lot of the code in the functions came from
{cheng2023sammed2d,
      title={SAM-Med2D}, 
      author={Junlong Cheng and Jin Ye and Zhongying Deng and Jianpin Chen and Tianbin Li and Haoyu Wang and Yanzhou Su and
              Ziyan Huang and Jilong Chen and Lei Jiangand Hui Sun and Junjun He and Shaoting Zhang and Min Zhu and Yu Qiao},
      year={2023},
      eprint={2308.16184},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""

def download_model():
    if os.path.exists('sam-med2d_b.pth') and os.path.getsize('sam-med2d_b.pth') >= 2.385*1024*1024*1024:
        # print("Model already exists locally. Using existing copy of it.")
        # print(os.path.getsize('sam-med2d_b.pth')/(1024*1024*1024), "GB")
        return
    # URL of the model
    url = "https://healthuniverse-models-production.s3.amazonaws.com/SAM-Med2D/data.pkl"

    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses

    # Get the total size of the file from the response headers (if present)
    total_size = int(response.headers.get('content-length', 0))

    # Setup the Streamlit progress bar
    # progress_bar = st.progress(0)
    # progress = 0

    # Save the content of the response to a local file
    with open('sam-med2d_b.pth', 'wb') as f:
        for chunk in response.iter_content(chunk_size=16*1024*1024):
            # progress += len(chunk)
            f.write(chunk)
            
            # Update the progress bar
            # progress_bar.progress(progress / total_size)

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
    download_model()
    model = sam_model_registry["vit_b"](_args).to(_args.device)
    model.eval()
    predictor = SammedPredictor(model)
    return predictor

# Initializaiton code to load the models.
#TODO: Make this thread safe.
lock = threading.Lock()
with lock:
    download_model()
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

    draw_points(selected_points, image_draw)

    image_pil = image_pil.convert('RGBA')
    image_pil.alpha_composite(mask_image)
    last_mask = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float, device=device))
    return [(image_pil, mask_image), last_mask]

def run_sammed_bbox(input_image, original_bboxes, last_mask, adapter_type):

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
    else:
        predictor = predictor_without_adapter
        
    image_pil = Image.fromarray(input_image).convert("RGBA")
    image = input_image
    H,W,_ = image.shape
    predictor.set_image(image)

    mask_image = Image.new('RGBA', (W, H), color=(0, 0, 0, 0))

    for i in range(len(original_bboxes)):
        last_mask_ = None
        # Making the model iterate on itself for slightly improved accuracy.
        for j in range(4):
            masks, _, logits = predictor.predict(
            box=original_bboxes[i], 
            mask_input = last_mask_, #TODO: Adjust this to reflect continuous improvement
            multimask_output=True 
            ) 
            last_mask_ = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float, device=device))
    
        # Draw masks on the image.
        mask_image_ = Image.new('RGBA', (W, H), color=(0, 0, 0, 0))
        mask_draw_ = ImageDraw.Draw(mask_image_)
        draw_mask(masks[0], mask_draw_, random_color=False)
        mask_image.alpha_composite(mask_image_)

    # image_pil = image_pil.convert('RGBA')
    image_pil.alpha_composite(mask_image)

    image_draw = ImageDraw.Draw(image_pil)
    for bbox in original_bboxes:
        upper_left_coordinate = (bbox[0], bbox[1])
        lower_right_coordinate = (bbox[2], bbox[3])
        width_ = max(2, W//300)
        image_draw.rectangle((upper_left_coordinate, lower_right_coordinate), outline="green", width=width_)

    return [(image_pil, mask_image), last_mask]


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
            0, 255), random.randint(0, 255), int(255*0.5))
    else:
        color = (30, 144, 255, int(255*0.5))

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
            original_point_x_coord, original_point_y_coord = point[0]/scaling_factor, point[1]/scaling_factor
            original_points.append(((original_point_x_coord, original_point_y_coord), label))
    except:
        st.warning("Points don't have a corresponding foreground/background label. This may impact the program's accuracy.")
        for point in resized_points:
            original_point_x_coord, original_point_y_coord = point[0]/scaling_factor, point[1]/scaling_factor
            original_points.append((original_point_x_coord, original_point_y_coord))
    return original_points
def get_original_bbox_coords(resized_bbox_coords, scaling_factor):
    """
    Given a list of coordinates on the resized image, get the original coordinates
    Input: the bbox coordinates on the resized image
    Output: the orignial bboxes rounded to the nearest integer.
    
    """
    resized_bbox_coords_numpy = np.array(resized_bbox_coords)
    original_bboxes = np.array(resized_bbox_coords)/scaling_factor
    return np.rint(original_bboxes)

def reset_stored_masks_and_images():
    st.session_state['last_mask'] = None
    if 'previous_image_with_mask' in st.session_state: 
        del st.session_state['previous_image_with_mask'] 
    if 'previous_mask' in st.session_state: 
        del st.session_state['previous_mask']
def initialize_styling():
    # TODO: image-with-the-indicated region segmented is the id of the subheader. if the subheader title changes, the id will change and some of the CSS below won't work.
    try:
        st.markdown("""
        <style>
        .stButton button {
            display: block;
            margin-left: auto;
            margin-right: auto;
            align-items: center !important;
        }
        #image-with-the-indicated-region-s-segmented {
            text-align: center;        
        }
        </style>
    """,unsafe_allow_html=True)
    except:
        st.warning("Centering of radio buttons isn't working. The app will still work, though the layout might be slightly off.")


def return_bbox_coordinates(canvas_result, stroke_width):
    """
    Returns the bounding box coordinates for objects drawn on a canvas.
    
    This function processes the drawing results from a canvas and calculates the 
    bounding box coordinates of each object, taking into consideration the stroke width.

    Parameters:
    - canvas_result (object): The result object from a Streamlit canvas which contains 
      drawing data.
    - stroke_width (int): The width of the stroke used to draw on the canvas. This 
      subtly influences the bounding box dimensions.

    Returns:
    - np.array: An array of bounding box coordinates. Each entry is a list in the format: 
      [x_upper_left, y_upper_left, x_lower_right, y_lower_right]. Returns an empty array 
      if no objects are detected.

    Notes:
    - The function assumes that the drawing consists of rectangles, and thus the bounding 
      box is determined by the top-left and bottom-right corners of each rectangle.
    """
    def get_bbox_info(left, top, width, height, stroke_width):
        """ 
        Calculate the bounding box coordinates for a given rectangle.

        Parameters:
        - left, top (float): The coordinates of the top-left corner of the rectangle.
        - width, height (float): The width and height of the rectangle.
        - stroke_width (int): The width of the stroke used to draw the rectangle.

        Returns:
        - list: A list containing the bounding box coordinates in the format: 
          [x_upper_left, y_upper_left, x_lower_right, y_lower_right].
        """
        x1, y1, x2, y2 = float(left), float(top), float(left + width + stroke_width), float(top + height + stroke_width)
        return [x1, y1, x2, y2]
    # Get the objects dataframe from the canvas result
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        for col in objects.select_dtypes(include=["object"]).columns:
            objects[col] = objects[col].astype("str")
        if objects.empty:
            return []
    bbox_info = objects.loc[:, ["left", "top", "width", "height"]]
    bbox_info = bbox_info.to_numpy()

    # Get the bbox coordinates in the format list([x_upper_left, y_upper_left, x_lower_right, y_lower_right])
    bbox_coordinates = []
    for row in bbox_info:
        bbox_coordinates.append(get_bbox_info(row[0], row[1], row[2], row[3], stroke_width))
    result = np.array(bbox_coordinates)
    return result

def return_clicked_coordinates(canvas_result, stroke_width):
    """
    Extracts the clicked coordinates and their associated labels from a canvas drawing.
    
    This function processes the drawing results from a canvas, identifies the clicked 
    points based on their fill colors, and returns their coordinates along with labels.

    Parameters:
    - canvas_result (object): The result object from a Streamlit canvas which contains 
      drawing data.
    - stroke_width (int): The width of the stroke used to mark points on the canvas. This 
      influences the coordinate extraction.

    Returns:
    - list of tuples: A list containing tuples, where each tuple consists of the point's 
      coordinates as a (x, y) pair and a label (1 for green, 0 for red). Returns an empty 
      list if no objects are detected.

    Notes:
    - The function assumes that points are marked with either green (indicating a FOREGROUND_POINT label) 
      or red (indicating a BACKGROUND_POINT label) fill colors.
    - Coordinates are adjusted based on stroke_width to better represent the true clicked 
      position.
    """
    # Get information abuot the points from the canvas result
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
    for col in objects.select_dtypes(include=["object"]).columns:
        objects[col] = objects[col].astype("str")
     
    if objects.empty:
        return []

    points_info = objects.loc[:, ["left", "top", "fill"]]
    points_info = points_info.to_numpy()

    # Compute the coordinates of the points and get the labels
    coords = []
    for row in points_info:
        x, y = int(row[0]) + int(stroke_width)//2, int(row[1])
        if row[2] == "green":
            label = FOREGROUND_POINT
        elif row[2] == "red":
            label = BACKGROUND_POINT
        coords.append(((round(x), round(y)), label))
    return coords
        

def main():
    initialize_styling()
    write_introduction()

    # Initialize variables and file uploading UI
    if 'last_mask' not in st.session_state:
        st.session_state['last_mask'] = None
    if 'run_id' not in st.session_state:
        st.session_state['run_id'] = 1
    if 'uploaded_file_id' not in st.session_state:
        st.session_state['uploaded_file_id'] = None
    if 'uploaded_image_URL' not in st.session_state:
        st.session_state['uploaded_image_URL'] = None

    # File uploader widget or get image from URL
    input_choice_help = """
    You can **upload an image** (.jpg, .jpeg, or .png) directly from your device or **type in the image URL**. \
    Select your preferred option.
    """
    input_choice = st.radio("You can ...", ["Upload an image (.jpg, .jpeg, or .png)", "Or, type in the image URL"], horizontal=True, help=input_choice_help)
    LOADING_IMAGE_MESSAGE = "Loading image and preparing to display it. This can take several seconds."
    # Handles case where user uploads image directly from their device.
    if "upload an image" in input_choice.lower():
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        uploading_file_progress_message = st.empty()
        container_1.empty()
        if uploaded_file is not None and uploaded_file.file_id != st.session_state['uploaded_file_id']:
            uploading_file_progress_message.info(LOADING_IMAGE_MESSAGE)
            reset_stored_masks_and_images()
            st.session_state['uploaded_file_id'] = uploaded_file.file_id
            st.session_state['uploaded_image_URL'] = None
    # Handles case where user enters the image URL.
    elif "url" in input_choice.lower():
            # Text input field where user can enter the image URL
            image_url_help = "Paste the direct URL link of the image you'd like to analyze. \
                For most images on the web, to get the URL, right-click on it. Then, select/click \
                on 'Copy Image Address'."
            image_url = st.text_input("Enter the URL of the image you want to upload below \
            (e.g., https://pressbooks.pub/app/uploads/sites/3987/2017/10/chest-case-12-2-909x1024.jpg)", 
            help = image_url_help)

            uploading_file_progress_message = st.empty()
            container_1.empty()
            if image_url.strip() != "": # Checks if the user entered some text for the image URL
                # Tries to retrieve image from the URL. If this doesn't work, it outputs an error message.
                # If it works, it handles this appropriately.
                try:
                    uploaded_file = io.BytesIO(urllib.request.urlopen(image_url).read())
                except Exception as e: 
                    st.error("The URL you entered isn't working. Or, you're facing connection issues.")
                    st.error("Error message: " + str(e)) # If the user enters an invalid URL
                    uploaded_file = None
                else:
                    if image_url != st.session_state['uploaded_image_URL']:
                        uploading_file_progress_message.info(LOADING_IMAGE_MESSAGE)
                        reset_stored_masks_and_images()
                        st.session_state['uploaded_image_URL'] = image_url
                        st.session_state['uploaded_file_id'] = [None]
            else:
                uploaded_file = None
            
    # Select ML model to use (with adapter or without one)
    model_selection_tooltip = """
    Choose the desired machine learning model from the dropdown menu. You can pick between "SAM-Med2D-B_w/o_adapter" and "SAM-Med2d-B".

    **Note**: When tested on various datasets, the "SAM-Med2D-B_w/o_adapter" model generally outperformed the "SAM-Med2D-B" model.
    """

    model = st.selectbox("Select Adapter for Model", ("SAM-Med2D-B_w/o_adapter", "SAM-Med2D-B"), help=model_selection_tooltip)


    tooltip_for_mode = """
    - In **Bounding Box** mode, you draw rectangular boxes around areas of the image you're interested in.
    - In **Multi-point interaction** mode, you directly interact with the image by marking specific points \
    on it; think of it like giving hints to the tool about which areas you're interested in and which areas you're not.

    **Note**: When tested on various datasets, \
    the **Bounding Box** mode somewhat outperformed the **Multi-point interaction mode**
    """
    mode = st.radio("Select mode", ["Bounding Box", "Multi-point interaction"], help=tooltip_for_mode, horizontal=True)
    st.divider()

    if uploaded_file is not None:
        with Image.open(uploaded_file) as img:
            img = img.convert("RGB")
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

                
                

            # Reset all previously marked points (and masks) if user clicks on button. Took out this due to canvas's inbuilt reset function. But, make sure that the masks are reset too.
    
            # Detect where the user clicked on the image and add 
            # the respective image coordinates to st.session_state and the image itself.
            if mode == "Multi-point interaction":
                stroke_width_ = 6
                st.markdown("""
                Below is your image. Please follow these steps:

                1. **Marking Points**:
                    - **Foreground Points (Green)**: Click on areas you're interested in. For instance, if you're examining an MRI scan and want to emphasize a potential lesion, place green points on that region.
                    - **Background Points (Red)**: Mark areas you're not focusing on. These red points help the tool differentiate areas of non-interest. For regions in the MRI scan you're not concerned about, mark them with red points.

                2. Use the buttons below the bottom left corner of the image to:
                    - **⤺ Undo** the last point.
                    - **⤼ Redo** a removed point.
                    - 🗑️ **Clear** all points.

                3. Once you're satisfied with your markings, click 'Run SAM-Med2D model' below the image.

                4. **Tips for Optimal Results**:
                    - A few strategically placed points often guide the model effectively. Less can be more.
                    - If the initial result isn't quite right:
                        1. Refine your selection by placing foreground points on overlooked areas.
                        2. Mark unwanted areas with background points.
                        3. Re-run the SAM-Med2D model.
                    - Repeating steps 1-3 multiple times can improve accuracy, as suggested by the authors \
                    of the [SAM-Med2D paper](https://arxiv.org/pdf/2308.16184.pdf). However, remember that \
                    perfection isn't always guaranteed.
                    - Sometimes, simply re-running the model 1-3 times without adding additional points can improve the accuracy.

                """)

                # Select whether the point is a foreground point (e.g., in the area the user is interested in) or a background point (outside the area in which the user is interested in)
                #TODO: (Minor) Consider adding the fact that the user can add more points and iteratively improve on the previous result.
                point_label_help = """
                - **Foreground Points (Green)**: Click on areas you're interested in. For instance, if you're examining an MRI scan and want to emphasize a potential lesion, place green points on that region.
                - **Background Points (Red)**: Mark areas you're not focusing on. These red points help the tool differentiate areas of non-interest. For regions in the MRI scan you're not concerned about, mark them with red points.  
                
                **Note**: Select the label before marking the point on the image.
                """
                point_label = st.radio("**Point Labels**", ["Foreground Point", "Background Point"], horizontal=True, help=point_label_help)
                if point_label == "Foreground Point":
                    label = FOREGROUND_POINT
                    fill_color_ = "green"
                    stroke_color_ = "green"
                elif point_label == "Background Point":
                    label = BACKGROUND_POINT
                    fill_color_ = "red"
                    stroke_color_ = "red"


                canvas_result = st_canvas(fill_color=fill_color_, stroke_color=stroke_color_, stroke_width= stroke_width_, 
                                            background_image=img, height=new_height, width=new_width, drawing_mode="point", 
                                            update_streamlit=True, key = "point_canvas")
                coords = return_clicked_coordinates(canvas_result, stroke_width_)
                #TODO: CheckS
                if len(coords) == 0:
                    st.session_state['last_mask'] = None
            elif mode == "Bounding Box":
                fill_color_ = "rgba(0, 255, 0, 0.2)"
                stroke_color_ = "rgba(0, 255, 0, 0.5)"
                st.markdown("""
                    Below is your image. Please follow these steps:

                    1. Draw rectangles around areas you're interested in.
                    2. Use the buttons below the bottom left corner of the image to:
                        - **⤺ Undo** the last rectangle.
                        - **⤼ Redo** a removed rectangle.
                        - 🗑️ **Clear** all rectangles.
                    3. Once done, click 'Run SAM-Med2D model' below the image.
                    """)
                try:
                    stroke_width_ = max(4, img.size[1]//200)
                except:
                    print("Img size dimensions error")
                    stroke_width_ = 4
                canvas_result =  st_canvas(fill_color=fill_color_, stroke_color=stroke_color_, stroke_width= stroke_width_, 
                                                            background_image=img, height=new_height, width=new_width, drawing_mode="rect", key="bbox_canvas")
                bounding_boxes = return_bbox_coordinates(canvas_result, stroke_width_)
                # st.write(bounding_boxes)
                if len(bounding_boxes) == 0:
                    st.session_state["last_mask"] = None

                uploading_file_progress_message.empty()
            # Run ML model on image with the points the user selected passed in.
            run_model = st.button("Run SAM-Med2D model.")
            if run_model:
                st.session_state['run_id'] += 1
                running_model = st.empty()
                running_model.info("Running the image segmentation algorithm. This can take several seconds.")
                if mode == "Multi-point interaction":
                    all_labels = [label_ for pt, label_ in coords]
                    if len(coords) == 0:
                        st.error("Please select at least one foreground point in the above image before running the model.\
                                To do this, click the 'Foreground Point' button and then click a point on the image.")
                    elif FOREGROUND_POINT not in all_labels:
                        st.error("Please select at least one foreground point in the above image before running the model.\
                                To do this, click the 'Foreground Point' button and then click a point on the image.")
                    else:
                        val1, val2 = run_sammed(np.array(img), get_original_points(coords, scaling_factor), # img = original image that wasn't modified or drawn upon
                                            st.session_state['last_mask'], model)
                        image_with_mask, mask = val1
                        st.session_state['last_mask'] = val2
                        st.divider()
                        st.subheader("Image with the Indicated Region(s) Segmented")
                        st.image(image_with_mask, use_column_width=True)

                        buf = io.BytesIO()
                        image_with_mask.save(buf, format="PNG")
                        image_bytes = buf.getvalue()
    
                        btn = st.download_button(
                            label="Download image",
                            data=image_bytes,
                            mime="image/png"
                        )
                if mode == "Bounding Box":
                    if len(bounding_boxes) == 0:
                        st.error("Please draw at least one rectangle in the above image before running the model.")
                    else:
                        val1, val2 = run_sammed_bbox(np.array(img), get_original_bbox_coords(bounding_boxes, scaling_factor), 
                                                    st.session_state['last_mask'], model)
                        image_with_mask, mask = val1
                        st.session_state['previous_image_with_mask'] = image_with_mask
                        st.session_state['previous_mask'] = mask
            if "previous_image_with_mask" in st.session_state and "previous_mask" in st.session_state:
                st.divider()
                st.subheader("Image with the Indicated Region(s) Segmented")
                image_with_mask = st.session_state["previous_image_with_mask"]
                st.image(image_with_mask, use_column_width=True)
                _, col1, col2 = st.columns([1, 3, 6])
                buf = io.BytesIO()
                image_with_mask.save(buf, format="PNG")
                image_bytes = buf.getvalue()

                img_download_btn_help = """
                Click here to download the image with the segmented parts in blue, \
                along with the corresponding bounding boxes or points.
                """
                btn = col1.download_button(
                    label="Download image",
                    data=image_bytes,
                    mime="image/png",
                    file_name="result_img_" + str(st.session_state['run_id']) + ".png",
                    help = img_download_btn_help
                )

                mask = st.session_state['previous_mask']
                buf_masks = io.BytesIO()
                mask.save(buf_masks, format="PNG")
                mask_bytes = buf_masks.getvalue()

                masks_download_btn_help = """
                Click here to download the masks only (without the underlying image). \
                The masks correspond to the blue parts of the image shown above.
                """
                btn_masks = col2.download_button(
                    label = "Download masks without underlying image",
                    data = mask_bytes,
                    mime = "image/png",
                    file_name="result_masks_" + str(st.session_state['run_id']) + ".png",
                    help = masks_download_btn_help
                )
            if run_model:
                running_model.empty()

            
if __name__ == "__main__": 
    main()