# Skin Disease Detection AI

An AI-based skin disease classification system using Deep Learning and Transfer Learning.

## Project Overview

This project uses a deep learning model to classify skin images into seven categories:

- Acne Mild
- Acne Moderate
- Acne Severe
- Eczema Mild
- Eczema Moderate
- Eczema Severe
- Normal

The trained model is integrated with a Streamlit web application where users can upload a skin image and receive a predicted class with confidence score.

## Technologies Used

- Python
- TensorFlow / Keras
- EfficientNetB0
- NumPy
- Pillow
- Scikit-learn
- Matplotlib
- Streamlit

## Model

The project uses EfficientNetB0 with transfer learning.

Input image size:

`224 × 224 × 3`

The final classification layer uses Softmax activation for 7 classes.

## Model Evaluation

The model was evaluated using a separate test dataset.

Final Test Accuracy:

`66.67%`

Evaluation metrics include:

- Precision
- Recall
- F1-score
- Confusion Matrix

The confusion matrix is available in:

`confusion_matrix.png`

## Web Application

The Streamlit application allows users to:

1. Upload a skin image
2. Preview the uploaded image
3. Run AI prediction
4. Display the predicted class
5. Display prediction confidence

## Project Structure

```text
SkinDiseaseAI/
│
├── app.py
├── train_v2.py
├── evaluation.py
├── predict.py
├── skin_disease_model_v2.keras
├── confusion_matrix.png
├── requirements.txt
└── README.md