# Ferrari Optimizer

## Overview
The **Ferrari Optimizer** is an experimental optimization algorithm designed to improve the training efficiency of machine learning models. This project demonstrates its use on a classifier trained with the Optical Recognition of Handwritten Digits dataset.

## Features
- Modular Multi-Layer Perceptron (MLP) architecture
- Custom optimization algorithm with adaptive learning rates
- Performance evaluation using accuracy, precision, recall, and F1-score
- Confusion matrix for detailed class-wise analysis

## Project Goals
The main objective of this project is to:
- Test the Ferrari Optimizer on a simple classification task
- Compare its performance to static learning rate optimization
- Identify the conditions where it excels or struggles

## Dataset
We use the [Optical Recognition of Handwritten Digits dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits). Each 32x32 bitmap is divided into non-overlapping 4x4 blocks, resulting in an 8x8 input matrix for the MLP.

## Implementation
1. **Data Processing:**
    - Shuffles and splits the dataset into training and validation sets
    - Supports plain-text and 1-hot encoded labels

2. **Network Architecture:**
    - Input layer: 64 neurons (8x8 matrix)
    - Hidden layer: 30 neurons
    - Output layer: 10 neurons (1-hot encoding)

3. **Training Process:**
    - Forward pass through the network
    - Loss calculated using Mean Squared Error (MSE)
    - Weights and biases updated using backpropagation

4. **Ferrari Optimizer:**
    - Adaptive learning rate updated based on the squared norm of the loss gradient
    - Stops optimization if the gradient norm falls below a threshold to prevent instability

## Observations
- The Ferrari Optimizer is effective for kickstarting the training process, especially with high initial losses
- Instability can occur as the learning rate may grow rapidly near stationary points
- A static learning rate generally outperforms the Ferrari Optimizer in this specific task
- Some classes (e.g., digits "1" and "3") show reduced recognition accuracy when using the Ferrari Optimizer due to unstable weight updates

## Conclusion
While the Ferrari Optimizer has potential, especially for early-stage training, a static learning rate approach proved more consistent in this context. Future improvements could include adaptive mechanisms to handle gradient instabilities more effectively.

## Authors
- Edoardo Caproni
- Tommaso Quintabà


