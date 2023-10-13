# SAM-Med2D Streamlit App

Welcome to the SAM-Med2D Streamlit App. This application provides an interactive interface for medical image segmentation based on the Segment Anything Model (SAM) model.

## Overview

SAM-Med2D offers a seamless user experience, allowing users to segment desired regions in medical images with impressive accuracy.

## Instructions for Using the App

### 1. Image Input
- **Upload an Image:** Directly from your device. Accepted formats include ".jpg", ".jpeg", and ".png".
- **Use Image URL:** If you're using Google Search, right-click on the image and select "Copy Image Address" to get the URL.

### 2. Model Selection
- Users have the option to select between two SAM-Med2D models:
  - `SAM-Med2D-B_w/o_adapter`
  - `SAM-Med2D-B`
  
  The 'SAM-Med2D-B_w/o_adapter' model has shown slight performance improvements over the other model in testing.

### 3. Segmentation Mode
- **Bounding Box Mode:** Draw rectangles around the region you wish to segment and then press 'Run SAM-Med2D model'.
- **Multi-point Interaction Mode:** Mark specific points on the image. Use Green for Foreground Points (e.g., potential lesions in an MRI scan) and Red for Background Points.

### 4. Actions Below Image
- ⤺ **Undo:** Removes the last point or bounding box.
- ⤼ **Redo:** Restores a removed point or bounding box.
- 🗑️ **Clear:** Erases all points or bounding boxes.

### 5. Initiating the Segmentation Process
- After marking the points or drawing the bounding box, click the "Run SAM-Med2D model" button.
- Wait for the algorithm to process the image. An info message will be displayed while the model is running.
- Ensure that you've selected a foreground point or drawn a bounding box, as prompted.

### 6. Viewing & Downloading Results
- Upon completion, the segmented image is displayed under "Image with the Indicated Region Segmented."
- Options to download the segmented image or just the masks without the base image are provided.

## Tips & Best Practices
- In **Multi-point Interaction** mode, strategic point placements can yield optimal results.
- For **Bounding Box** mode, ensure the boxes encompass the entire region of interest.

## Conclusion
We hope that you find this app helpful in your medical image analysis tasks. We're committed to refining and enhancing this tool and welcome all feedback to serve our users better. Once you're ready, navigate to the "SAM-Med2D App" page in the sidebar to begin.

## Credits
This app is grounded on the SAM-Med2D paper's model. We thank the authors Junlong Cheng, Jin Ye, Zhongying Deng, and many others for their commendable work.

Also, a significant portion of the code in the app comes from the paper's associated [GitHub repository](https://github.com/OpenGVLab/SAM-Med2D).
