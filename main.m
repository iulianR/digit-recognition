clear; close all; clc;

% choice = menu('Pick option', ...
%               'Train Neural Network using the MNIST training set', ...
%               'Manually test images from the MNIST test set', ...
%               'Automatically test all images from the MNIST test set');

fprintf('Pick one option:\n\n');
fprintf('1. Train Neural Network using the MNIST training set\n');
fprintf('2. Test the accuracy of the Neural Network on the MNIST test set\n');
fprintf('3. Test images taken from video camera\n');
fprintf('4. Test images taken from filel\n');

input('Choice: ')

if ans == 1
    trainNeuralNetwork()
else
    testset = (ans == 2);
    camera = (ans == 3);
    file = (ans == 4);
    filename = '';

    if file == 1
        input('Input filename (with extension): ', 's')
        filename = ans;
    end

    fprintf('\nUse provided weights or weights generated by training the network? \n')
    fprintf('NOTE: Select option 2 only after you successfully finished the training \n')
    fprintf('1. Provided\n');
    fprintf('2. Trained\n\n');

    input('Choice: ')

    provided = (ans == 1);

    testNeuralNetwork(provided, testset, camera, file, filename);
end