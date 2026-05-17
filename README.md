# Image Forgery Detection System  
### Dual-Branch Deep Learning Model using EfficientNetV2S + Error Level Analysis (ELA)

---

## Badges

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Keras](https://img.shields.io/badge/Keras-NeuralNetwork-red)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-green)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![License](https://img.shields.io/badge/License-OpenSource-lightgrey)

---

## Overview

This project is an AI-based image forgery detection system designed to classify images into Original or Tampered categories.  
It leverages Deep Learning (EfficientNetV2S) combined with forensic Error Level Analysis (ELA) for improved robustness and accuracy.

---

## Problem Statement

Detecting manipulated images has become a major challenge due to advanced editing tools.  
This system aims to automatically identify whether an image is authentic or tampered using deep learning and forensic techniques.

---

## Dataset

CASIA 2.0 Image Tampering Dataset

- Authentic Images (Au)  
- Tampered Images (Tp)  

Split:
- Train / Validation / Test  

Dataset Link:
https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset

---

## Methodology

### Preprocessing
- Image resizing (224×224)
- Data augmentation
- Class balancing
- Corrupted image removal

### Error Level Analysis (ELA)
- Recompression of images
- Pixel difference extraction
- Enhancement of tampering traces

---

## Model Architecture

### CNN Baseline Model
- Convolutional layers
- Batch Normalization
- Dense classifier

### Final Model (Dual-Branch EfficientNetV2S)
- RGB Branch → EfficientNetV2S
- ELA Branch → EfficientNetV2S
- Feature Fusion (Concatenation)
- Fully connected classification head

---

## Training Strategy

Phase 1:
- Freeze backbone
- Train classification head

Phase 2:
- Unfreeze top layers
- Fine-tune model

Techniques:
- Focal Loss
- Data Augmentation
- Class Weights

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## Results

The dual-branch model significantly outperformed the CNN baseline in detecting tampered images and generalizing to unseen data.

---

## Model Files (Google Drive)

Due to large size, models are stored externally:

- CNN Model: https://drive.google.com/file/d/1KA8hJU4Uu7to2f2Jy9DDVnnnVbvNQ1p6/view?usp=sharing  
- EfficientNetV2S Model: https://drive.google.com/file/d/1iuxWnmO479Rw27ypsHFNM5ojTdusnhVR/view?usp=drive_link
- Training CNN History:https://drive.google.com/file/d/1wQieMU4c1uyLsgDnNLKOFtg1xG9y_ASO/view?usp=drive_link 
- Training EfficientNetV2S History:https://drive.google.com/file/d/1SgG7yQbgjkHRAst0cRk7W4891JWEn2SC/view?usp=drive_link 

---

## Web Application

Streamlit-based real-time inference system.

Features:
- Upload image
- Predict authenticity
- Show confidence score
- Display ELA visualization

Run:
streamlit run app.py

---

## Project Structure

Image-Forgery-Detection/
├── app.py
├── requirements.txt
├── README.md
├── notebooks/
└── images/

---

## Technologies Used

Python, TensorFlow, Keras, EfficientNetV2S, OpenCV, PIL, scikit-learn, Streamlit

---

## GitHub Profile Style

### About
This project demonstrates skills in Deep Learning, Computer Vision, and Image Forensics.

### Key Highlights
- Dual-Branch Architecture
- ELA + Deep Learning Fusion
- Transfer Learning
- End-to-End Deployment

### Future Work
- Transformer models
- Tampering localization
- Cloud deployment
- API integration

---

## Author

AI Project  
Image Forensics & Deep Learning System
