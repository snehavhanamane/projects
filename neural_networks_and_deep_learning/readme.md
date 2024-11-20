# CIFAR-10 Image Classification

This project focuses on building a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into one of 10 classes. The model is implemented in PyTorch, trained on the CIFAR-10 dataset, and evaluated based on its accuracy on the test set.

## Project Overview
- **Dataset**: CIFAR-10 (60,000 images in 10 classes)
- **Model**: Custom CNN architecture with multiple BackboneBlocks
- **Training**: 30 epochs using cross-entropy loss and the Adam optimizer
- **Evaluation**: Achieved 76.8% accuracy on the test set

## Features
- **Data Preparation**: Applied normalization and data augmentation using `torchvision.transforms`.
- **Model Architecture**: Backbone-based CNN with configurable parameters (`N`, `k`, and `num_filters`).
- **Training and Evaluation**: Tracked loss and accuracy for both training and test sets across epochs.
- **Results Analysis**:
  - Per-class accuracy for detailed insights.
  - Plots showing the evolution of loss and accuracy over epochs.

## Results
- **Overall Test Accuracy**: 76.8%
- **Per-Class Accuracies**:
  - Highest: `Truck` (86%)


## Requirements
- Python 3.8+
- Libraries listed in `requirements.txt`

## Usage
1. Clone the repository:
   
   git clone https://github.com/username/cifar10-classification.git

## Install dependencies:

pip install -r requirements.txt

## Train and evaluate the model:

python train.py
(Optional) Visualize results in Jupyter Notebook:

jupyter notebook cifar10_analysis.ipynb

## Project Structure
README.md: Project documentation.

requirements.txt: Python dependencies.

train.py: Main script for training and evaluation.

cifar10_analysis.ipynb: Notebook for visualizing results and predictions.

models/: Contains model architecture files.

data/: Directory for dataset (downloaded automatically via torchvision).

## Dataset
The CIFAR-10 dataset is a widely used benchmark dataset available via PyTorch's torchvision.datasets. It includes 60,000 images (32x32 pixels) in 10 classes:

Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck