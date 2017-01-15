function [J grad] = costFunction(nn_params, ...
                                 input_layer_size, ...
                                 hidden_layer_size, ...
                                 num_labels, ...
                                 X, y, lambda)
%COSTFUNCTION Implements the cost function for a two layer neural network which
% performs classification

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight
% matrices for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Number of training examples
m = size(X, 1);

% The first step is to compute the cost (without regularization)

% Add the a vector of bias units to our data set and that should give us
% the activation of the units in the first layer (or the input layer)
a1 = [ones(m, 1) X]; % (60000, 784 + 1)

% Compute the activation of units in the layer 2 (or the hidden layer).
% We use the sigmoid function as our activation function.
z2 = a1 * Theta1'; % (60000, 785) x (785, 35) = (60000, 35)
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1) a2]; % (60000, 36)

% Compute the activation of units in layer 3 (or the output layer). This is also
% equivalent to our hypothesis function, h.
z3 = a2 * Theta2'; % (60000, 36) x (36, 10) = (60000, 10)
a3 = sigmoid(z3);
h = a3;

yvec = zeros(m, num_labels);
for i = 1:m
    yvec(i, y(i)) = 1; % (60000, 10)
end

% We compute the cost function J for our neural network based on the formula
t1 = -yvec .* log(h); % (60000, 10) .x (60000, 10)
t2 = (1 - yvec) .* log(1 - h); % (60000, 10) .x (60000, 10)
J = 1/m * sum(sum(t1 - t2));

% We add regularization for our cost function
sumTheta1 = sum(sum(Theta1(:, 2:end) .^ 2));
sumTheta2 = sum(sum(Theta2(:, 2:end) .^ 2));
regTerm = lambda / (2 * m) * (sumTheta1 + sumTheta2);
J += regTerm;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% The second step is to implement backpropagation to compute the gradient for
% the neural network cost function
for t = 1:m
    % Set the input layer's values to the t-th training example and perform
    % a forward pass for layers 2 and 3
    a1 = [1; X(t,:)'];
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

    % See which class the current training example belongs to
    yv = ([1:num_labels] == y(t))';

    % Perform backpropagation
    delta3 = a3 - yv;
    delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)];
    delta2 = delta2(2:end);

    % Accumulate gradients
    Theta1_grad += delta2 * a1';
    Theta2_grad += delta3 * a2';
end

% Divide by m to obtain the unregularized gradient
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% Add regularization to the gradient
regTermBackProp = lambda / m * [zeros(size(Theta1, 1), 1), Theta1(:,2:end)];
Theta1_grad += regTermBackProp;
regTermBackProp = lambda / m * [zeros(size(Theta2, 1), 1), Theta2(:,2:end)];
Theta2_grad += regTermBackProp;

% Unroll gradients
grad = [Theta1_grad(:); Theta2_grad(:)];

end
