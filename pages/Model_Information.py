import streamlit as st
import streamlit.components.v1 as components
from instructions import write_introduction

text = """
Across the nine MICCAI2023 datasets, SAM-Med2D (with the adapter layer dropped during test time)
scored a Dice score of 90.12% (given BBox prompts) and 83.41% (given single-point prompts), compared to 81.93% and 60.31% 
respectively for the SAM-Med2D model (with the adapter during test time) and 85.35% and 48.08% respectively for the SAM model.



"""

import requests
from PIL import Image
from io import BytesIO

def app():
    st.title('SAM-Med2D Model Information')
    st.subheader("👉 Source of Model")
    st.write("The model is from the [SAM-Med2D paper](https://arxiv.org/pdf/2308.16184.pdf). The authors are Junlong Cheng, Jin Ye, \
          Zhongying Deng, Jianpin Chen, Tianbin Li, Haoyu Wang, Yanzhou Su, Ziyan Huang, Jilong Chen, Lei Jiangand, \
          Hui Sun, Junjun He, Shaoting Zhang, Min Zhu, and Yu Qiao.")
    st.write("The images and much of the text on this page are from the corresponding [GitHub repository's](https://github.com/OpenGVLab/SAM-Med2D) README.md file.")
    

    st.subheader('👉 Dataset')
    st.markdown("""
    SAM-Med2D is trained and tested on a dataset that includes **4.6M images** and **19.7M masks**. 
    This dataset covers 10 medical data modalities, 4 anatomical structures + lesions, and 31 major human organs. 
    This is likely the largest and most diverse medical image segmentation dataset so far in terms of quantity and coverage of categories.
    """)
    st.image("assets/dataset.png", caption='Dataset Visualization', use_column_width=True)

    st.subheader('👉 Framework')
    st.markdown("""
    The pipeline of SAM-Med2D. We freeze the image encoder and incorporate learnable adapter layers in each Transformer block to acquire domain-specific knowledge in the medical field. 
    We fine-tune the prompt encoder using point, Bbox, and mask information, while updating the parameters of the mask decoder through interactive training.
    """)
    st.image("assets/framwork.png", caption='Framework Visualization', use_column_width=True)

    st.subheader('👉 Results')
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
    <p>Note that the performance metric is the Dice score, which quantifies the overlap between the area that's segmented and the ground truth (i.e., correct answer).</p>
    """, scrolling = True)

    st.subheader('👉 Visualization')
    st.image("assets/visualization.png", caption='Model Visualization', use_column_width=True)

    # st.markdown("[Link to the official paper](https://arxiv.org/abs/2308.16184)")
    # st.markdown("[Link to the GitHub repository](https://github.com/OpenGVLab/SAM-Med2D)")

if __name__ == "__main__":
    app()

def model_accuracy():
    st.write("""
    SAM Med2D's performance at a resolution of $256\times256$ showcases its impressive capabilities (the percentages are the model's Dice scores):
    - **Bounding Box Prompt**: 79.30%
    - **1 Point Prompt**: 70.01%
    - **3 Points Prompt**: 76.35%
    - **5 Points Prompt*: 78.68%

    This robustness is further reflected in its generalization validation on the MICCAI2023 datasets. The model, even without the adapter layer (SAM-Med2D*), consistently outperforms the base SAM model across multiple benchmarks.
    """)

def model_information():
    write_introduction()