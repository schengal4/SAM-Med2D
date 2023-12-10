import streamlit as st
import io

def write_introduction():
    st.title("SAM-Med2D Streamlit App")
    st.write("SAM-Med2D is an interactive medical image segmentation model based on the Segment Anything Model (SAM) model. \
             This app provides a seamless experience for users to segment desired regions in medical images with good accuracy.")
def write_app_instructions():
    st.subheader("Instructions for Using the App:")

    # Instructions for image input
    st.markdown("""
    1. **Image Input:**
        - You can either: 
            * **Upload an image** directly from your device. Accepted formats are ".jpg", ".jpeg", or ".png".
            * Or **type in the image URL**. If using Google Search, right-click on the image and select "Copy Image Address" to obtain the URL.
        """)

    # Instructions for model selection
    st.markdown("""
    2. **Model Selection:**
       - You'll have the option to select between two SAM-Med2D models: `SAM-Med2D-B_w/o_adapter` and `SAM-Med2D-B`. Make your choice from the dropdown menu. \
        The 'SAM-Med2D-B_w/o_adapter' model slightly outperforms the other according to testing done by the model makers.
    """)

    # Segmentation mode instructions
    st.markdown("""
    3. **Segmentation Mode**
        - **Bounding Box Mode**: Draw rectangles around the area you wish to segment.
            - Once you've drawn the desired bounding boxes, press 'Run SAM-Med2D model'.
        - **Multi-point Interaction Mode**: Mark specific points on the image.
            * Foreground Points (Green): Indicate areas of interest. For instance, potential lesions in an MRI scan.
            * Background Points (Red): Specify areas you're not interested in, helping the model differentiate.
            * After marking the points, click 'Run SAM-Med2D model'.
    """)

    # How to undo/redo/clear the previously drawn point/bounding box.
    st.markdown("""
    4. Buttons Below Image:
        - ⤺ Undo: Removes the last point or bounding box.
        - ⤼ Redo: Restores a removed point or bounding box.
        - 🗑️ Clear: Erases all points or bounding boxes.
    """)

    # Initiating Segmentation Process Instructions
    st.markdown("""
    5. **Initiating the Segmentation Process:**
       - After marking your points, click the "Run SAM-Med2D model" button.
       - Please be patient, as the algorithm processes the image. An info message will appear indicating that the model is running.
       - If you haven't selected a foreground point or drawn a bounding box (depending on the mode), you'll be prompted to do so before the model can proceed.
    """)
    
    # Viewing and Downloading Results from ML algorithm
    st.markdown("""
    6. **Viewing & Downloading the Results:**
       - Once segmentation is complete, the segmented image will be displayed under the subheader "Image with the Indicated Region Segmented."
       - To download the segmented image, click the "Download image" button.
       - To download the masks alone without the underlying image, click the "Download masks without underlying image" button.
    """)

    # Tips & Best Practices section
    st.subheader("Tips & Best Practices:")
    st.write("""
    - For the **Multi-point Interaction** mode, just a few well-placed points can often provide the clarity the model needs.
    - For the **Bounding Box** mode, ensure the boxes encapsulate the entirety of the region you're interested in.
    """)

    # Conclusion
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
                    
if __name__ == "__main__": 
    write_introduction()
    write_app_instructions()
    cite_the_source("### Credits:")
