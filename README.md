# Object Recognition in Images with CIFAR-10

This project aims to perform object recognition in images using the CIFAR-10 dataset. The goal is to train a Convolutional Neural Network (CNN) and Random Forest machine learning model on the CIFAR-10 dataset and compare their performance in recognizing objects in images.

## Data Preprocessing
- Load the CIFAR-10 dataset into the environment
  - Import the CIFAR-10 dataset (if not already available in the environment)
- Preprocess the data by converting the images into a format suitable for training the model
- Split the dataset into training and testing sets

## Convolutional Neural Network (CNN)
- Train a CNN model on the training set and evaluate its performance on the test set
- Use appropriate techniques to improve the performance of the model (e.g. data augmentation, early stopping, etc.)
- Plot the training and validation loss and accuracy to understand the model's behavior

## Random Forest
- Train a Random Forest model on the training set and evaluate its performance on the test set
- Use appropriate techniques to improve the performance of the model (e.g. hyperparameter tuning, feature selection, etc.)
- Plot the feature importances to understand the relative importance of each feature

## Results
- Compare the performance of both models (CNN and Random Forest) and discuss their pros and cons
- Report the final accuracy, precision, recall, and other performance metrics of both models
- Provide insights on why one model performed better than the other

Libraries used:
- Tensorflow
- Keras
- Numpy
- Sklearn
- Matplotlib

## Running the Code
- The code is implemented using Python.
- Make sure you have the required libraries installed.
- Clone the repository to your local machine.
- Run the Python script to see the output.
