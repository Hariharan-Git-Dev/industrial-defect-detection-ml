# industrial-defect-detection-ml
Classical machine learning pipeline for steel surface defect detection using real industrial datasets, feature engineering, and SVM classification.

Industrial Steel Surface Defect Detection (Classical ML)
 Project Overview

This project implements an industrial steel surface defect detection system using classical computer vision and machine learning techniques.
The goal is to classify steel surface images as Defective or Good (Non-defective) based on handcrafted visual features.

Unlike deep-learning approaches, this project emphasizes:

Interpretability

Low computational cost

Practical industrial deployment scenarios

Problem Statement

In steel manufacturing, surface defects such as scratches, cracks, or inclusions can significantly impact product quality. Manual inspection is slow and error-prone.

This project automates image-level defect detection, enabling:

Faster inspection

Consistent quality control

Reduced human dependency

Dataset

Source: Severstal Steel Defect Detection (Kaggle)

Type: Real industrial grayscale images

Classes:

Good (defect-free images extracted using CSV annotations)

Defective (images containing one or more defects)

Dataset Preparation

Defect-free images were identified by parsing train.csv

Class imbalance was handled by creating a balanced dataset

Raw dataset files are not included in this repository (intentionally)

Project Structure
Defect_Detection_ML/
│
├── extract_clean_images.py     # Extracts defect-free images from CSV
├── create_balanced_dataset.py  # Balances good vs defective samples
├── extract_features.py         # Feature extraction pipeline
├── train_model.py              # Model training & evaluation
├── demo_predict.py             # Single image prediction demo
├── .gitignore                  # Excludes datasets & models
└── README.md

Feature Engineering

The model relies on handcrafted visual features commonly used in industrial inspection:

Edge Density (Canny edge detection)

Contour Statistics

Number of contours

Average contour area

Intensity Variance

Texture Features

Local Binary Patterns (LBP)

Gradient Features

Histogram of Oriented Gradients (HOG)

All images are:

Converted to grayscale

Resized to a fixed resolution

Noise-reduced using Gaussian blur

Machine Learning Model

Task: Binary Classification (Good vs Defective)

Approach: Classical ML (no deep learning)

Training: Stratified train-test split

Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Sample Results
Accuracy ≈ 67%

Confusion Matrix:
[[284 101]
 [138 197]]


Accuracy is intentionally reported honestly.
The focus is robust feature engineering, not inflated metrics.

How to Run
1️ Install Dependencies
pip install numpy pandas opencv-python scikit-image scikit-learn

2️ Train the Model
python train_model.py

3️ Predict on a New Image
Place an image and run:

python demo_predict.py








Place an image and run:

python demo_predict.py
