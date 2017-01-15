function testNeuralNetwork(provided=1, testset=1, camera=0, file=0, filename='')
%TESTNEURALNETWORK Tests the neural network and displays accuracy

% Load images and labels
X = loadMNISTImages('t10k-images.idx3-ubyte');
y = loadMNISTLabels('t10k-labels.idx1-ubyte');
y(y == 0) = 10;

% Get the number of test examples
m = size(X, 1);

input_layer_size  = 784; % 28x28 Input Images of Digits
hidden_layer_size = 35;  % 35 hidden units
num_labels        = 10;  % 10 labels

if provided == 0
    load('trained.mat')
else
    load('nn_params.mat');
end

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

pred = predict(Theta1, Theta2, X);

fprintf('Training Set Accuracy (percentage of correct guesses): %f\n\n', mean(double(pred == y)) * 100);

if testset == 1
    %  Randomly permute examples
    rp = randperm(m);
    for i = 1:m
        % Display
        fprintf('Displaying Example Image\n');
        displayData(X(rp(i), :));

        pred = predict(Theta1, Theta2, X(rp(i),:));
        fprintf('Neural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));

        % Pause
        fprintf('Press enter to continue.\n\n');
        pause;
    end
elseif camera == 1
    pkg load image-acquisition
    obj = videoinput('v4l2', '/dev/video0');
    set(obj, 'VideoFormat', 'RGB3');
    set(obj, 'VideoResolution', [640 480]);
    start(obj, 1)
    while true
        img = getsnapshot(obj);
        figure(1)
        image(img)
        figure(2)
        pred = predict(Theta1, Theta2, imageTo28x28Gray(img));
        fprintf('Neural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
        pause
        fprintf('Press enter to get current frame.\n')
    end
elseif file == 1
    img = imread('digit.png');
    figure(1)
    image(img);
    figure(2)
    pred = predict(Theta1, Theta2, imageTo28x28Gray(img));
    fprintf('Neural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
end

end