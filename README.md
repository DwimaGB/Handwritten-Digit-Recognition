# Handwritten Digit Recognition with CNN

This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits based on the MNIST dataset. The model is trained to classify images of digits (0-9) with high accuracy. Follow the steps below to set up the project, preprocess data, train the model, and test it with new or test data.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Training](#2-model-training)
  - [3. Model Testing with Test Data](#3-model-testing-with-test-data)
  - [4. Testing with New Data](#4-testing-with-new-data)
- [Requirements](#requirements)

## Features
- CNN model for accurate handwritten digit classification
- Data preprocessing script for transforming raw data
- Model testing with new images or existing test data
- Simple interface for drawing and testing handwritten digits

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Handwritten-Digit-Recognition.git
   cd Handwritten-Digit-Recognition

2. **Install the required packages**:
    ```bash
      pip install -r requirements.txt

## Usage 

1. **Data Preprocessing:** Run the `data_preprocessing.py` script to preprocess the MNIST dataset into a suitable format for training:

    ```bash
    python data_preprocessing.py
    ```

2. **Model Training:** Train the model on the preprocessed data by running:

    ```bash
    python model_training.py
    ```

3. **Model Testing with Test Data:** Evaluate the trained model's performance on the test dataset by executing:

    ```bash
    python model_testing.py
    ```

4. **Testing with New Data:** Use the `digit_drawer.py` script to open a simple GUI for testing new handwritten digit inputs:

    ```bash
    python digit_drawer.py
    ```


## Requirements
- Python 3.7+
- TensorFlow
- Pandas
- Numpy
- Matplotlib (for digit drawer visualization)
