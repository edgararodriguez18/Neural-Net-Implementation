# Neural-Net-Implementation

This project's objective was to create a neural network model that can correctly classify the XOR problem, the sklearn, and MNIST handwritten digits problem within a minimal error tolerance. This Python program creates an Artificial Neural Network from scratch without the use of a library like keras tensorflow. The user can edit and select an arbitrary number of layers, the number of nodes in each layer, the number of training epochs, the Stochastic Gradient Descent with momentum parameter, select an activation function for each layer (sigmoid and ReLU), and the loss error function used for evaluation (Mean Square Error or Categorical Cross
Entropy).

To run this project, select which of the three wrapper files to run and edit the input parameters as needed.

## Short Description of each file:

- mnistwrapper.py: wrapper that is used to run the MNIST handwritten digits classification problem.
- sklearnwrapper.py: wrapper that is used to run the sklearn handwritten digits classification problem.
- xorwrapper.py:  wrapper that is used to run the XOR classification problem.
- neuralnet.py: The main python file containing the artificial neural network implementation including the feed forward, Backpropagation, and weight update process.

The wrappers used for the XOR, sklearn, and MNIST handwritten digits datasets utilize the
python interface that can be used to initialize and create the multilayer perceptron, train the
dataset for X number of epochs, incorporating an early stopping function, plot the loss error function
graph, and do output evaluation metrics using sklearn.metrics (accuracy score, confusion matrix,
classification report).
