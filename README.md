# An ensemble of U-Net models to segment TB consistent lesions in frontal chest radiographs

# Abstract
Automated semantic segmentation of Tuberculosis (TB)-consistent lesions in chest X-rays (CXRs) using deep learning (DL) methods would help reduce radiologist effort, supplement clinical decision-making, and improve patient care. Existing literature discusses training and evaluating these models using coarse bounding box annotations to semantically segment TB-consistent lesions. This results in including a considerable fraction of false positives and negatives in the annotations that may adversely impact segmentation performance. In this study, we propose to evaluate the combined benefits of using fine-grained annotations of TB-consistent lesions, and train and construct ensembles of the variants of U-Net models for segmenting TB-consistent lesions in the original and bone-suppressed frontal chest X-rays (CXRs). We evaluate segmentation performance using several ensemble methods such as bitwise AND, bitwise-OR, bitwise-MAX and stacking. We observed that the segmentation performance achieved using the stacking ensemble demonstrated superior performance compared to the constituent models and other ensemble methods. To our best knowledge, this is the first study to apply ensemble learning to improve fine-grained TB-consistent lesion segmentation performance.  

# Process pipeline

![Process pipeline](figure_1.png width="512" height="512")


# Requirements
h5py==3.1.0
imageio==2.11.1
matplotlib==3.5.1
numpy==1.19.5
opencv_python==4.5.4.58
pandas==1.3.4
Pillow==9.1.1
scikit_image==0.18.3
scikit_learn==1.1.1
scipy==1.7.2
segmentation_models==1.0.1
skimage==0.0
tensorflow==2.6.2
tqdm==4.62.3

# Code description
The python code unet_ensemble.py contains the following:
Loading libraries; loading and creating data numpy arrays; preprocessing functions; loss functions; evaluation metrics; boundary uncertainty evaluation; model loading, compiling, and training; model inference; constructing ensembles using bitwise operations; constructing a stacking ensemble. 


