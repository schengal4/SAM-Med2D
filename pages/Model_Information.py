import streamlit as st
import streamlit.components.v1 as components

def app():
    st.title('SAM-Med2D Model Information')
    
    st.subheader('👉 What is SAM-Med2D?')
    st.markdown("""
    SAM-Med2D is a state-of-the-art tool tailor-made for medical image analysis. All you need to do \
    is show the model the areas of interest – either by clicking on the image or \
    drawing a box around them. SAM-Med2D then zeroes in on those regions for deeper insights. \
    Curious about the scientific foundation behind it? Dive into the \
    [SAM-Med2D paper](https://arxiv.org/pdf/2308.16184.pdf).

    The innovation comes from a team of dedicated researchers who combined their knowledge \
    to advance the field of medical imaging. To learn more about the technical details, \
    you can explore their [GitHub repository](https://github.com/OpenGVLab/SAM-Med2D). \
    The images, captions, and some of the text on this page come from the repository's \
    README.md file or the SAM-Med2D paper.
                """)
    st.markdown("""
    SAM-Med2D is a cutting-edge tool designed to analyze medical images; users can tell the model which regions they're interested in \
    (by clicking on points in those regions or drawing boxes around them) and the model identifies those exact regions.\
    This tool is built based on a research study presented in the [SAM-Med2D paper](https://arxiv.org/pdf/2308.16184.pdf). 
                
    The innovation comes from a team of dedicated researchers who combined their knowledge \
    to advance the field of medical imaging. To learn more about the technical details, \
    you can explore their [GitHub repository](https://github.com/OpenGVLab/SAM-Med2D). \
    The images, captions, and some of the text on this page come from the corresponding \
                GitHub repository's README.md file.
    """
    )

    
    st.subheader('👉 Dataset information')
    st.markdown("""
    The dataset SAM-Med2D is trained on has 4.6 million medical images, covering 10 different medical imaging \
    modalities, 31 major organs, and their corresponding anatomical structures.
    """)

    image_caption = """
    Overview of the dataset used in this study. (a) A total of 31 major organs, along with
    their corresponding anatomical structures, with an asterisk (*) denoting the presence of lesion labels
    within the dataset. (b) The distribution of medical imaging modalities along with their corresponding proportions in
    the dataset are presented (scaled logarithmically). (c) The number of images and masks categorized
    by anatomical structure, along with the total count encompassing the dataset.
    """
    st.image("assets/dataset.png", caption=image_caption, use_column_width=True)

    st.subheader("👉 A Peek Into the Model's Brain")
    st.write("""
    SAM-Med2D's 'brain' is a network of algorithms that help it process images. It has several layers \
    and parts that work together to accurately segment the image (i.e., the user can select a \
    point or draw a box on the image to tell the model what area they're interested in and the \
    model will output the part(s) of interest within that area).
    """)
    framework_image_caption = """
    The pipeline/architecture of SAM-Med2D. The image encoder is frozen and \
    learnable adapter layers are incorporated \
    in each Transformer block to acquire domain-specific knowledge in the medical field.
    The prompt encoder was fine-tuned using point, Bbox, and mask information, while the
    parameters of the mask decoder were updated through interactive training.
    """
    st.image("assets/framework.png", caption=framework_image_caption, use_column_width=True)

    st.subheader('👉 Model Accuracy/Performance')

    st.markdown("""
    SAM Med2D's performance at a resolution of 256 **x** 256 demonstrates its strong capabilities \
    ; the percentages are the model's Dice scores, which quantifies the overlap \
    between the area that's segmented and the ground truth (i.e., correct answer)). \
    A score of 100% means perfect overlap, and 0% means no overlap at all:
    - **Bounding Box Prompt**: 79.30%
    - **1 Point Prompt**: 70.01%
    - **3 Points Prompt**: 76.35%
    - **5 Points Prompt**: 78.68%

    This robustness is further reflected in its generalization validation on the MICCAI2023 datasets. The model, even without the adapter layer (SAM-Med2D*), consistently outperforms the base SAM model across multiple benchmarks.
    """)

    components.html("""
    <table>
        <caption align="center">Generalization validation on 9 MICCAI2023 datasets, where "*" denotes that we drop adapter layer of SAM-Med2D in test phase. </caption>
    <thead>
        <tr>
        <th rowspan="2">Datasets</th>
        <th colspan="3">Bbox prompt (%)</th>
        <th colspan="3">1 point prompt (%)</th>
        </tr>
        <tr>
        <th>SAM</th>
        <th>SAM-Med2D</th>
        <th>SAM-Med2D*</th>
        <th>SAM</th>
        <th>SAM-Med2D</th>
        <th>SAM-Med2D*</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td align="center"><a href="https://www.synapse.org/#!Synapse:syn51236108/wiki/621615">CrossMoDA23</a></td>
        <td align="center">78.98</td>
        <td align="center">70.51</td>
        <td align="center">84.62</td>
        <td align="center">18.49</td>
        <td align="center">46.08</td>
        <td align="center">73.98</td>
        </tr>
        <tr>
        <td align="center"><a href="https://kits-challenge.org/kits23/">KiTS23</a></td>
        <td align="center">84.80</td>
        <td align="center">76.32</td>
        <td align="center">87.93</td>
        <td align="center">38.93</td>
        <td align="center">48.81</td>
        <td align="center">79.87</td>
        </tr>
        <tr>
        <td align="center"><a href="https://codalab.lisn.upsaclay.fr/competitions/12239#learn_the_details">FLARE23</a></td>
        <td align="center">86.11</td>
        <td align="center">83.51</td>
        <td align="center">90.95</td>
        <td align="center">51.05</td>
        <td align="center">62.86</td>
        <td align="center">85.10</td>
        </tr>
        <tr>
        <td align="center"><a href="https://atlas-challenge.u-bourgogne.fr/">ATLAS2023</a></td>
        <td align="center">82.98</td>
        <td align="center">73.70</td>
        <td align="center">86.56</td>
        <td align="center">46.89</td>
        <td align="center">34.72</td>
        <td align="center">70.42</td>
        </tr>
        <tr>
        <td align="center"><a href="https://multicenteraorta.grand-challenge.org/">SEG2023</a></td>
        <td align="center">75.98</td>
        <td align="center">68.02</td>
        <td align="center">84.31</td>
        <td align="center">11.75</td>
        <td align="center">48.05</td>
        <td align="center">69.85</td>
        </tr>
        <tr>
        <td align="center"><a href="https://lnq2023.grand-challenge.org/lnq2023/">LNQ2023</a></td>
        <td align="center">72.31</td>
        <td align="center">63.84</td>
        <td align="center">81.33</td>
        <td align="center">3.81</td>
        <td align="center">44.81</td>
        <td align="center">59.84</td>
        </tr>
        <tr>
        <td align="center"><a href="https://codalab.lisn.upsaclay.fr/competitions/9804">CAS2023</a></td>
        <td align="center">52.34</td>
        <td align="center">46.11</td>
        <td align="center">60.38</td>
        <td align="center">0.45</td>
        <td align="center">28.79</td>
        <td align="center">15.19</td>
        </tr>
        <tr>
        <td align="center"><a href="https://tdsc-abus2023.grand-challenge.org/Dataset/">TDSC-ABUS2023</a></td>
        <td align="center">71.66</td>
        <td align="center">64.65</td>
        <td align="center">76.65</td>
        <td align="center">12.11</td>
        <td align="center">35.99</td>
        <td align="center">61.84</td>
        </tr>
        <tr>
        <td align="center"><a href="https://toothfairy.grand-challenge.org/toothfairy/">ToothFairy2023</a></td>
        <td align="center">65.86</td>
        <td align="center">57.45</td>
        <td align="center">75.29</td>
        <td align="center">1.01</td>
        <td align="center">32.12</td>
        <td align="center">47.32</td>
        </tr>
        <tr>
        <td align="center">Weighted sum</td>
        <td align="center">85.35</td>
        <td align="center">81.93</td>
        <td align="center">90.12</td>
        <td align="center">48.08</td>
        <td align="center">60.31</td>
        <td align="center">83.41</td>
        </tr>
    </tbody>
    </table>
    <p>The numbers you see are Dice scores, which indicate that SAM-Med2D generally performs well; the higher the better. 
    </p>
    """, scrolling = True)

    st.subheader('👉 Visualization')
    model_visualization_caption = """
    Qualitative comparisons were made between the segmentation results of SAM-Med2D and
    SAM. SAM-Med2D outperformed SAM significantly.
    """
    st.image("assets/visualization.png", caption=model_visualization_caption, use_column_width=True)

if __name__ == "__main__":
    app()
