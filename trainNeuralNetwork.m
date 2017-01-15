function trainNeuralNetwork()
%TRAINNEURALNETWORK Implements backpropagation and gradient descent to train a
% two layers neural network

% Load MNIST images
X = loadMNISTImages('train-images.idx3-ubyte');
y = loadMNISTLabels('train-labels.idx1-ubyte');
y(y == 0) = 10;

% Get the number of examples in the training set
m = size(X, 1)

input_layer_size  = 784; % 25x25 Input Images of Digits
hidden_layer_size = 35;	 % 35 hidden units
num_labels = 10;		 % 10 labels

% Initialize weights to random values
initial_Theta1 = initializeWeights(input_layer_size, hidden_layer_size); % 35 x 785
initial_Theta2 = initializeWeights(hidden_layer_size, num_labels); % 10 x 36

% Unroll parameters into vector
initial_nn_params = [initial_Theta1(:); initial_Theta2(:)]; % 27835 x 1

options = optimset('MaxIter', 50);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) costFunction(p, ...
                                 input_layer_size, ...
                                 hidden_layer_size, ...
                                 num_labels, X, y, 1);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

save('trained.mat', 'nn_params');

end
