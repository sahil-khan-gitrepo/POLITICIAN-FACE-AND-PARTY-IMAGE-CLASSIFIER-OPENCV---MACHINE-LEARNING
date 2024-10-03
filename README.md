# Politician Face and Party Image Classifier

## Project Overview
This project is focused on building a **Politician Face and Party Image Classifier** using **OpenCV** for image processing and **Machine Learning** algorithms to predict the political party affiliation of various politicians based on their images. The model aims to detect faces in politician images, process the images to extract features, and classify them into different political parties using various classifiers.

The project involves:
- Face detection using **OpenCV** and **Haar Cascades**.
- Feature extraction using **Wavelet Transform**.
- Training machine learning classifiers (**SVM, Random Forest, Logistic Regression**) to predict party affiliation.
- Achieving an accuracy of **88%** in classifying politicians based on their party.

## Table of Contents
1. [Project Motivation](#project-motivation)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Project Workflow](#project-workflow)
    - Data Preprocessing
    - Face Detection
    - Feature Extraction
    - Model Training
    - Evaluation
5. [Results](#results)

## Project Motivation
Political party identification is crucial in media analysis and voter education. Automated systems that recognize politicians and their political affiliations help the voter know which party a particular leader belongs to. The project has potential applications in journalism, social media analysis, and public discourse monitoring.

In this project, we focus on developing a model that automatically detects faces of politicians from images, extracts meaningful features, and predicts their party affiliation with high accuracy. The project serves as an example of how machine learning can be applied to face detection.

## Dataset
The dataset used in this project consists of:
- Politician images from various sources.
- Party affiliation labels corresponding to each politician.
  
Each image contains the face of a politician, and the task is to detect the face, extract features, and classify the image into the correct political party.

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - **OpenCV**: For face detection and image preprocessing.
  - **Scikit-Learn**: For training machine learning models (SVM, Random Forest, Logistic Regression).
  - **Pandas and Numpy**: For data manipulation and analysis.
  - **Matplotlib**: For visualization of results.
  - **Joblib**: For model deployment and saving trained models.

## Project Workflow

### 1. Data Preprocessing
- Collected images of politicians and labeled them with their corresponding party affiliations by storing them in their respective folders.
  
### 2. Face Detection
- Used **OpenCV's Haar Cascade** classifier to detect faces in the politician images.
- Haar Cascades work by detecting features like edges and textures in an image, and they are widely used for real-time face detection.
- The face detection model cropped the region containing the face from the original image, reducing noise from irrelevant parts of the image.

### 3. Feature Extraction
- Applied **Wavelet Transform** to extract key features from the detected face images.
- Wavelet Transform is a powerful tool for feature extraction as it allows us to decompose images into different frequency components, highlighting important details.
  
### 4. Model Training
- Trained three different machine learning classifiers:
  - **Support Vector Machine (SVM)**: A powerful classifier that works well for image classification.
  - **Random Forest**: An ensemble method that builds multiple decision trees and averages their predictions.
  - **Logistic Regression**: A linear model that works well for binary or multi-class classification.
- Used **Scikit-learn** to train the models and **Grid Search** to fine-tune hyperparameters.

### 5. Evaluation
- Evaluated the model using metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.
- The Random Forest classifier achieved the best accuracy of **88%** on the test dataset, indicating that the model could correctly predict the political party of a politician with high confidence.

## Results
- The project achieved an **88% accuracy** on test data, with SVM, Random Forest, and Logistic Regression models being trained.
- The **Random Forest** model outperformed other classifiers, making it the most reliable for real-world use.
- The model can be deployed for real-time predictions, making it suitable for use in media analysis applications.

### Model Performance:

| Classifier        | Accuracy |
|-------------------|----------|
| Support Vector Machine (SVM) | 86%      |
| Random Forest     | 88%      |
| Logistic Regression | 84%    |


