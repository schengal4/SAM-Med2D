import streamlit as st
import io
# Change file name to Instructions.py
def write_introduction():
    st.title("SAM-Med2D Streamlit App")
    st.write("SAM-Med2D is an interactive medical image segmentation model based on the Segment Anything Model (SAM) model. \
             This app provides a seamless experience for users to segment desired regions in medical images with good accuracy.")
def write_app_instructions():
    st.subheader("Instructions for Using the App:")

    st.markdown("""
    1. **Image Input:**
        - On the main screen, you'll be presented with two options: 
            * Upload an image (.jpg, .jpeg, or .png) directly from your device.
            * Enter the URL of an image.
        - Select your preferred option. 
    """)
    st.markdown(""" 
        *If Uploading Directly:*
        - Click on "Choose an image..." and navigate to your desired image.       
                
        *If Using an Image URL:*
        - Paste the image's URL in the provided text box. 
            * Tip: If sourcing an image from Google Search, right-click on the image and select "Copy Image Address" to obtain its URL.
        """)
    st.markdown("""
    2. **Model Selection:**
       - You'll have the option to select between two SAM-Med2D models: `SAM-Med2D-B_w/o_adapter` and `SAM-Med2D-B`. Make your choice from the dropdown menu.
    """)
    st.markdown("""
    3. **Marking Points on the Image:**
       - Once your image is displayed, decide if you want to mark a "Foreground Point" (a point in the region you aim to segment) or a "Background Point" (areas you want to exclude from the segmentation).
       - Click on the corresponding radio button under "Point Labels".
       - Click directly on the image to place your chosen point. 
    """)

    st.markdown("""
    4. **Editing Your Points:**
       - To erase all marked points, click the "Erase All Marked Points on the Image" button.
    """)

    # ... [Your existing code for Editing Your Points]

    st.markdown("""
    5. **Initiating the Segmentation Process:**
       - After marking your points, click the "Run SAM-Med2D model" button.
       - Please be patient, as the algorithm processes the image. An info message will appear indicating that the model is running.
       - If you haven't selected a foreground point, you'll be prompted to do so before the model can proceed.
    """)

    # ... [Your existing code for Initiating the Segmentation Process]

    st.markdown("""
    6. **Viewing & Downloading the Results:**
       - Once segmentation is complete, the segmented image will be displayed under the subheader "Image with the Indicated Region Segmented."
       - To download the segmented image, click the "Download image" button.
    """)

    # ... [Your existing code for Viewing & Downloading the Results]

    # ----- Tips & Best Practices -----
    st.subheader("Tips & Best Practices:")
    st.write("""
    - Ensure that the points you mark on the image are representative of the regions you wish to segment.
    - For the best results, mark multiple points both inside and outside the region of interest.
    """)

    # ----- Conclusion -----
    st.subheader("Conclusion:")
    st.write("""
    We hope that you find this app helpful in your medical image analysis tasks. We're committed to refining and enhancing this tool and welcome all feedback to serve our users better.
    """)
    st.write("""
    Once you've read the instructions, navigate to the "SAM-Med2D App" page from the sidebar to use the application.\n
    """)
    

def cite_the_source(section_header):
    st.write(section_header)
    st.write("This app is built on the model from the SAM-Med2D paper. The authors are Junlong Cheng, Jin Ye, \
          Zhongying Deng, Jianpin Chen, Tianbin Li, Haoyu Wang, Yanzhou Su, Ziyan Huang, Jilong Chen, Lei Jiangand, \
          Hui Sun, Junjun He, Shaoting Zhang, Min Zhu, and Yu Qiao.")

    st.write("Also, a significant portion of the code in the app comes from the paper's associated GitHub code: https://github.com/OpenGVLab/SAM-Med2D.")

# Initialize variables and file uploading UI
                    
if __name__ == "__main__": 
    write_introduction()
    write_app_instructions()
    cite_the_source("### Credits:")
