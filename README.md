# Debiasing Random Forest on CMNIST

## Overview

This project investigates the bias in simple machine learning models, specifically Random Forests, using a variant of the MNIST dataset called CMNIST. In CMNIST, each digit is associated with a color, and a strong correlation (95%) is introduced between digit and color (aligned), with the remaining 5% being conflicting (misaligned). The goal is to study how Random Forests perform on both aligned and conflicting samples, and to explore whether bias is inherent to the classifier or influenced by the data representation.

## Project Roadmap

1. **Dataset Preparation**:  
   - Images are loaded from disk using OpenCV, flattened, and split into training, validation, and testing sets.
   - Labels (`Y`) represent the digit class, while `Y_bias` encodes the color (bias) information, both extracted from the filename.

2. **Model Training**:  
   - A Random Forest classifier is trained on the flattened image data.
   - The model is evaluated on both aligned and conflicting samples for each class.

3. **Feature Representation**:  
   - Features extracted from a pre-trained neural network are also used to test whether bias is due to the classifier or the data representation.

4. **Evaluation**:  
   - Performance is measured separately for aligned and conflicting samples, across all classes.
   - Visualizations are provided to illustrate the dataset and model predictions.

## Dataset Structure

- The dataset is organized into folders for training (`align/`, `conflict/`), validation (`valid/`), and testing (`test/`).
- Each image filename follows the format `x_y_z.png`:
  - `x`: image ID
  - `y`: digit label
  - `z`: color (bias) label

## Usage

1. **Requirements**:
   - Python 3.11+
   - numpy, opencv-python, matplotlib, seaborn, scikit-learn

2. **Running the Notebook**:
   - Open `RF_bias.ipynb` in Jupyter or VSCode.
   - Execute the cells sequentially to load data, train the model, and view results.

3. **Customization**:
   - You can modify the dataset path or experiment with different Random Forest parameters and feature representations.

## Key Findings

- The notebook provides a detailed analysis of how Random Forests can inherit or mitigate dataset bias.
- It also explores the impact of learned feature representations on model bias.

## Visualizations

- The notebook includes visualizations of both aligned and conflicting samples, as well as model performance metrics.
