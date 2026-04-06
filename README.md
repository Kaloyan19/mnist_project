# MNIST Digit Classification with PyTorch

A clean and well-structured PyTorch project for handwritten digit recognition using the MNIST dataset. The project implements and compares two neural network architectures: **Multi-Layer Perceptron (MLP)** and **Convolutional Neural Network (CNN)**.

## Project Overview

- **Dataset**: MNIST (60,000 training + 10,000 test images of handwritten digits 0-9)
- **Models**:
  - **BetterMLP**: Fully connected neural network with 3 hidden layers (512-256-128) + Dropout
  - **SimpleCNN**: Convolutional neural network with 2 Conv layers + MaxPooling (achieves higher accuracy)
- **Final Results**:
  - MLP: **98.41%** test accuracy
  - CNN: **99.19%** test accuracy

## Repository Structure
mnist_project/
├── models/                    # Saved model weights (.pth files)
├── results/                   # Generated plots and visualizations
├── mnist_photos/              # Clean MNIST-style test images
├── my_handwritten_edited_photos/ # My own handwritten digits
├── model.py                   # Shared BetterMLP model definition
├── mnist_2.py                 # Training script for MLP
├── cnn_model.py               # Training script for CNN
├── visualizations.py          # Loss & Accuracy curves
├── results_visualization.py   # Confusion Matrix + Error Examples
├── comparison.py              # MLP vs CNN comparison plot
├── predict_my_digits_mnist_2.py        # Test on clean MNIST images (MLP)
├── predict_my_handwriting_mnist_2.py   # Test on my handwriting (MLP)
├── predict_my_handwriting_cnn.py       # Test on my handwriting (CNN)
└── README.md


## Features

- Clean code structure with separated model definition
- Training with Adam optimizer and CrossEntropyLoss
- Comprehensive visualizations (loss curves, accuracy curves, confusion matrix, error examples)
- Comparison between MLP and CNN
- Prediction scripts for both standard MNIST images and real handwritten digits

## How to Run

1. Clone the repository
2. Install requirements:
   ```bash
   pip install torch torchvision torchaudio matplotlib seaborn scikit-learn pillow numpy

3. Train the models:
python mnist_2.py      # Train MLP
python cnn_model.py    # Train CNN

4. Generate visualizations:
python visualizations.py
python results_visualization.py
python comparison.py

## Technologies Used
Python
PyTorch
torchvision
Matplotlib + Seaborn
PIL (Pillow) for image preprocessing