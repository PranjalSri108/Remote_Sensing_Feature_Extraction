# Satellite Image Feature Extraction with Improved DeepLabv3+
## Project Overview

This project implements an advanced satellite image segmentation model using an improved DeepLabv3+ architecture. It's designed to classify and extract various land features and man-made structures from high-resolution satellite imagery.

![alt text](https://github.com/PranjalSri108/Satellite_Image_Feature_Extraction_with_Improved_DeepLabv3/blob/main/img.jpg?raw=true)

## Techniques Used

1. Deep Learning
2. Semantic Segmentation
3. Transfer Learning
4. Data Augmentation (Patching, rotation, flipping, color jittering)
5. Atrous Spatial Pyramid Pooling (ASPP)
6. Channel Attention (Squeeze-and-Excitation block)
7. Multi-scale Feature Extraction

## Unique Selling Points (USP)

- Improved DeepLabv3+ architecture with channel attention for enhanced accuracy
- Combined loss function (Dice Loss and Soft Cross-Entropy Loss) for robust training
- Custom IoU metric for accurate evaluation
- Efficient data handling with patching technique
- Web application for user-friendly interaction and feature selection
- Specialized for remote sensing applications
- End-to-end solution from data preprocessing to visualization

## Architecture

The model uses an improved DeepLabv3+ architecture with the following components:

1. Backbone: ResNet101 (pre-trained on ImageNet)
2. Enhanced Atrous Spatial Pyramid Pooling (ASPP) module with 2x2 atrous convolutions
3. Channel attention module (Squeeze-and-Excitation block)
4. Decoder with refinement stages
5. Output layer for multi-class segmentation

## Implementation Details

1. Data Preprocessing:
   - Load and resize images to a fixed size
   - Create 256x256 pixel patches from large satellite images
   - Apply data augmentation techniques

2. Model:
   - Custom ImprovedDeepLabv3Plus class
   - Enhanced ASPP module for multi-scale feature extraction
   - Channel attention mechanism
   - Decoder for upsampling and refinement

3. Training:
   - Combined loss function (Dice Loss and Soft Cross-Entropy Loss)
   - Adam optimizer with learning rate of 0.0001
   - Custom IoU metric for evaluation

4. Inference:
   - Patch-based processing of large satellite images
   - Post-processing for visualization

5. Web Application:
   - Streamlit-based user interface
   - Feature selection/deselection option
   - Display of segmentation masks and bounding boxes

## Results

The model outperforms standard architectures in terms of mIoU (mean Intersection over Union) and loss metrics. Visualizations of segmentation results are available through the web application.

## Research and References

This project is based on extensive research in deep learning for remote sensing. Key references include:

- Zhang, H., et al. (2020). "Application of Deep Learning in Remote Sensing Image Processing."
- Lee, J. G., et al. (2017). "Deep Learning for Feature Extraction in Remote Sensing: A Case-Study of Aerial Scene Classification."
- Anderson, P., & White, N. (2022). "Post-processing Techniques for Visualizing Segmentation Outputs."
