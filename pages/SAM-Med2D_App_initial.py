import requests
import streamlit as st
import time

# Create a function for the download logic
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = Namespace()
args.device = device
args.image_size = 256
args.encoder_adapter = True
args.sam_checkpoint = "pretrain_model/sam-med2d_b.pth"
predictor_with_adapter = load_model(args) # Loads the model with the adapter layer
args.encoder_adapter = False
predictor_without_adapter = load_model(args)
def download_model(chunk_size):
    # URL of the model
    url = "https://healthuniverse-models-production.s3.amazonaws.com/SAM-Med2D/data.pkl"

    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses

    # Get the total size of the file from the response headers (if present)
    total_size = int(response.headers.get('content-length', 0))

    # Setup the Streamlit progress bar
    progress_bar = st.progress(0)
    progress = 0

    # Save the content of the response to a local file
    with open('sam-med2d_p.pth', 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            progress += len(chunk)
            f.write(chunk)
            
            # Update the progress bar
            progress_bar.progress(progress / total_size)

    st.write(f"Download complete for chunk_size {chunk_size}!")

  # Change/add sizes as needed

download_model(16*1024*1024)

r1 = """
Chunk size: 1048576 - Average Time: 273.53186384836835 seconds

Chunk size: 8388608 - Average Time: 87.38633855183919 seconds

Chunk size: 33554432 - Average Time: 112.93882918357849 seconds

Chunk size: 67108864 - Average Time: 132.00136637687683 seconds
===

Chunk size: 1048576 - Average Time: 111.43516000111897 seconds

Chunk size: 8388608 - Average Time: 96.28959902127583 seconds

Chunk size: 33554432 - Average Time: 214.71106402079263 seconds

Chunk size: 67108864 - Average Time: 198.31816983222961 seconds

===

Chunk size: 2097152 - Average Time: 84.24789269765218 seconds

Chunk size: 4194304 - Average Time: 91.77218985557556 seconds

Chunk size: 8388608 - Average Time: 87.55327184995015 seconds

Chunk size: 16777216 - Average Time: 81.2591290473938 seconds

====

Chunk size: 2097152 - Average Time: 81.67152333259583 seconds

Chunk size: 4194304 - Average Time: 75.16103251775105 seconds

Chunk size: 8388608 - Average Time: 184.37408939997354 seconds

Chunk size: 16777216 - Average Time: 87.16078511873881 seconds

"""
