# Machine-Learning-Algorithms
 # 1. Linear Regression
Compares the implementation of the Pocket Algorithm and Linear Regression using Python and NumPy with their counterparts in scikit-learn. It would demonstrate the fundamental understanding of the algorithms by coding them from scratch and then evaluate the performance against the established modules in scikit-learn.

## Pocket Algorithm with NumPy vs SciKit
The task requires the creation of four key functions: fit_Perceptron for training the algorithm, errorPer for calculating the error rate, pred for making predictions, and confMatrix for evaluating the performance through a confusion matrix. These functions are designed to work together to identify the optimal linear boundary that separates two classes of data points, demonstrating the fundamental concepts of machine learning and algorithmic implementation.

 The test_SciKit function takes training and test datasets, along with their respective labels, to train a Perceptron model and output a confusion matrix, providing a clear comparison of model performance against a custom NumPy implementation. This segment underlines the practical application of scikit-learn's Perceptron and metrics modules.

## Linear Regression with NumPy vs SciKit
The Linear Regression implementation in Python using NumPy, focusing on solving the least squares problem. It will describe the fit_LinRegr function for computing regression coefficients, the mse function for mean squared error calculation, and the pred function for making predictions. The document will highlight the use of NumPy functions like shape, hstack, ones, dot, and linalg.inv and will also reference the subsetFn utility for cases with non-full-rank matrices, providing a comprehensive guide to linear regression analysis.

The implementation of Linear Regression using scikit-learn in Python will detail the test_SciKit function, which takes in training and test datasets to train a regression model, and computes mean squared error using scikit-learn's metrics library.

# 2. Feed Forward Neural Network

## Data Preparation

The Iris dataset, traditionally a multi-class dataset, is modified to serve as a binary classification problem. Class labels are transformed such that the class '1' is labeled as '-1' and class '2' as '1', while class '0' is discarded. The data is then split into training and test datasets with an 80/20 ratio. This preprocessed data serves as the input for both the custom neural network and the Scikit-learn MLPClassifier.

## Neural Network Implementation

### Training (fit_NeuralNetwork)
A neural network is initialized with random weights and trained over a specified number of epochs. The training involves the following steps:

- Initialization: Weights are initialized for input, hidden, and output layers according to the provided architecture.
- Forward Propagation (forwardPropagation): The network processes the input by moving it through the layers, applying weights, and using ReLU activation functions for hidden layers and a sigmoid function for the output layer.
- Error Computation (errorPerSample): For each training sample, the error is computed using the log loss error function.
- Backpropagation (backPropagation): The error is backpropagated through the network to compute gradients with respect to the weights.
- Weight Update (updateWeights): Weights are updated using the computed gradients and a learning rate (alpha).

After each epoch, the average error across all samples is recorded.

### Activation Functions
- ReLU (activation, derivativeActivation): The ReLU function is applied to neuron activations, and its derivative is calculated.
- Sigmoid (outputf, derivativeOutput): The sigmoid function is used at the output layer to map activations to probabilities, and its derivative is calculated for backpropagation.

### Error Functions
- Log Loss Error (errorf, derivativeError): The error function and its derivative for a binary classification are implemented.
### Prediction and Evaluation
- Prediction (pred): The function predicts class labels for the test data using the trained weights.
- Confusion Matrix (confMatrix): A confusion matrix is generated to evaluate the predictions against the actual labels.
### Error Visualization
- Error Plot (plotErr): The function plots the training error as it decreases over epochs.
- Scikit-Learn MLPClassifier (test_SciKit)

The program also demonstrates the usage of Scikit-learn's MLPClassifier to train a neural network. It is trained on the same dataset, and the performance is evaluated using a confusion matrix and accuracy scores for both training and testing sets.

### How to Run

Make sure to have Python installed with the following packages: numpy, seaborn, matplotlib, scikit-learn. The program can be run as a script. The custom neural network will train, and its performance will be visualized alongside the performance of the Scikit-learn MLPClassifier. Conversly, upload the NeuralNetwork-2.ipynb notebook onto Google Colab to run the program.

# 3. Fully Connected Neural Network (FNN)

## Overview

The Python script trains a neural network on the notMNIST dataset and includes functionality for experimenting with different learning rates to compare their effects on training. It performs the following steps:

- Data Loading: Custom function loadData imports the dataset, normalizes the images, and splits it into training, validation, and test sets.
- Dataset Class: notMNIST is a subclass of torch.utils.data.Dataset tailored to handle the notMNIST dataset.
- Neural Network Definition: FNN is a torch.nn.Module subclass defining a feedforward neural network with three linear layers.
- Training and Evaluation: Functions train and get_accuracy perform training and evaluate the model, respectively.
- Experimentation: Function experiment runs the training process with specified hyperparameters and returns the trained model and accuracy history.
- Comparison of Learning Rates: Function compare_lr runs experiments for different learning rates and plots their training and test accuracies.
## Detailed Description

### Data Preprocessing
- The data is loaded from an '.npz' file, shuffled, and split into 10,000 training samples, 6,000 validation samples, and the remaining samples for testing.
- The pixel values are scaled to the [0, 1] range for model input.
### Neural Network Model (FNN Class)
- The FNN class initializes a three-layer neural network with ReLU activations for hidden layers.
- It accepts input images of size 28x28, flattens them, and processes through two hidden layers before outputting to 10 classes corresponding to the 10 different letters.
### Training Procedure
- The train function executes the forward pass, loss computation, backpropagation, and optimizer steps for each epoch and batch.
- Model performance (accuracy) on the training, validation, and test sets is calculated after each epoch.
- Cross-entropy loss and AdamW optimizer are used for training.
### Experiment and Learning Rate Comparison
- ' experiment ' function performs a full training and evaluation cycle using the specified learning rate and the number of epochs.
compare_lr runs multiple experiments for different learning rates, records the accuracy history, and visualizes the results, allowing comparison to see how learning rate affects the learning process.
### Running the Script
- The main execution calls compare_lr with a list of learning rates and runs for a default of 50 epochs.
- Training progress is printed out for each learning rate, and a plot is generated to compare train and test accuracies for each learning rate.
### Visualization
- After training, accuracy trends are plotted to compare the impact of different learning rates on the model's performance.
<img width="722" alt="image" src="https://github.com/aritrachoudhury/Machine-Learning-Algorithms/assets/97995705/c0b97e2f-1e93-408f-8836-25951141dfb2">
<img width="717" alt="image" src="https://github.com/aritrachoudhury/Machine-Learning-Algorithms/assets/97995705/dbcc0610-d368-4d93-b0a6-480ff59b1ff9">
<img width="729" alt="image" src="https://github.com/aritrachoudhury/Machine-Learning-Algorithms/assets/97995705/864202d0-9967-4909-9532-b8fc0e66dbf1">
<img width="730" alt="image" src="https://github.com/aritrachoudhury/Machine-Learning-Algorithms/assets/97995705/4b0a08db-ced7-47e4-8fe4-353ff161bc52">

# 4. K-Means and Mixture of Gaussian Models (MoG)
## K-Means Clustering Overview

This project demonstrates the KMeans clustering process using both the PyTorch framework for a more "from scratch" approach and the Scikit-learn library for a more straightforward, out-of-the-box solution.

### Features 
- Load and Split Data: The load_data function loads a 2D dataset and splits it into training and validation sets, preparing the data for clustering.
- PyTorch Clustering Implementation: Detailed implementation of KMeans using PyTorch, including the steps for initializing cluster centers, assigning data points to the nearest cluster, and updating cluster centers.
- Scikit-learn Clustering Implementation: Utilizes Scikit-learn's KMeans class to perform clustering, serving as a baseline comparison to the PyTorch implementation.
- Evaluation and Visualization: Both implementations include functions to evaluate the clustering performance and visualize the results, making it easier to analyze and compare outcomes.

### Implementation Details

#### PyTorch KMeans
The PyTorch implementation (train_kmean_torch) involves manually defining the cluster centers and optimizing them using gradient descent with the Adam optimizer. The training process iteratively updates the cluster centers based on the mean squared distance to the data points.

#### Scikit-learn KMeans
The Scikit-learn version uses the library's KMeans class, which abstracts away the details of initializing centers, assigning clusters, and updating centers. It's a simple yet powerful tool for quick clustering analysis.

## Gaussian Mixture Models Overview

Presents a comprehensive implementation of Gaussian Mixture Models (GMM) using PyTorch. The code is designed to offer insights into how Gaussian Mixture Models can be implemented and optimized using gradient descent. By leveraging PyTorch's automatic differentiation, this implementation allows for an exploration of GMMs beyond the conventional scope provided by libraries like Scikit-learn. The project showcases how to initialize model parameters, compute the Expectation step in GMMs, and perform clustering on 2D datasets.

### Features

- PyTorch-Based Implementation: Utilizes PyTorch for all aspects of GMM training, including parameter initialization, probability density function calculations, and optimization.
- Custom Initialization Options: Supports initializing GMM parameters randomly or using KMeans clustering for potentially improved convergence.
- Comprehensive GMM Functions: Implements functions to compute the log probability density function (PDF) of Gaussian distributions, log posterior probabilities, and the overall training loop.
- Visualization: Includes functionality to visualize the clustering results, providing an intuitive understanding of how GMMs group data and position their means and variances.
- Evaluation on Test Data: Demonstrates how to evaluate the trained GMM on a separate test dataset, calculating the mean log-likelihood of the test data under the model.

### Implementation Details

The core of this project is the train_gmm function, which iteratively updates GMM parameters to maximize the log-likelihood of the data. This process involves:

- Parameter Initialization: Depending on the choice, parameters can be initialized randomly or based on the centroids found by KMeans clustering.
- Log Probability Density Function Calculation: Computes the log PDF of the data points under each Gaussian component.
- Log Posterior Calculation: Calculates the log posterior probabilities of each component given the data.
- Optimization: Uses PyTorch's Adam optimizer to adjust the model parameters based on the gradient of the log-likelihood.

### Dataset
Before running the code, make sure you have a 2D dataset named data2D.npy in your working directory. This dataset is used for both training and validating the clustering models. If you wish to use a different dataset, adjust the load_data function accordingly.




