# Handwritten Digit Recognition using a Neural Network

This is a implementation of a neural network able to recognize handwritten digits. More exactly, it's a 2-layer perceptron as it uses the feedforward propagation and the backpropagation algorithms.

## Included files
* `main.m` - start point of the applicaiton
* `trainNeuralNetwork.m` - function used to train the neural network on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). When the operation finishes with success, a set of weights will be saved in a newly created `trained.mat` file
* `testNeuralNetwork.m` - allows testing the neural network
* `costFunction.m` - implements the neural network cost function for a two layer neural network which performs classification
* `predict.m` - predict the label of an input given a trained neural network using the feedforward propagation algorithm
* `sigmoid.m` - sigmoid function used for the activation of network's units
* `sigmoidGradient.m` - derivative of the sigmoid function
* `initializeWeights.m` - randomly initialize weights to break symmetry. Used the method described here [Random initialization](https://web.stanford.edu/class/ee373b/nninitialization.pdf)
* `imageTo28x28Gray.m` - adapted from the resources section of a [Coursera Machine Learning course](https://www.coursera.org/learn/machine-learning/home/welcome)
* `nn_params.mat` - weights from a previous train of the model
* `loadMNISTimages.m`, `loadMNISTlabes.m` - helper functions for extracting images and labels from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Source: [Stanford ULFDL wiki](http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset)
* `train-images.idx3-ubyte`, `train-images.idx1-ubyte` - training set images and labels from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
* `t10k-images.idx3-ubyte`, `t10k-images.idx1-ubyte` - training set images and labels from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/)

## Installation
### 1. Install Octave
#### Debian/Ubuntu
    # apt-get install octave
#### Fedora
    # dnf install octave

### 2. Install dependencies
These are required by the image-acquisition package (step 3).
#### Debian/Ubuntu
    # apt-get install liboctave-dev libv4l-dev libfltk1.3-dev
#### Fedora
    # dnf install octave-devel libv4l-devel fltk-devel


#### 3. Install the image-acquisition package from Octave forge
This package is required if you want to classify images from your webcam's video stream.

    pkg install -forge image-acquisition

## Documentation

TODO