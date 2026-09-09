% Step-by-step RBF classification example.
% This small example uses already extracted numeric features.
% It explains the RBF classifier before working with handwritten images.
clear; clc; close all;

% Two-dimensional toy features for four objects.
P = [0.0 0.1 0.9 1.0;
     0.0 0.1 1.0 0.9];

% Two classes: 0 and 1. Convert labels to one-hot targets.
labels = [0 0 1 1];
T = [labels == 0; labels == 1];

% Choose one center for each class.
C = [0.05 0.95;
     0.05 0.95];
width = 0.3;

% 1. Calculate the response of each RBF neuron.
distance = P - C(:,1);
F1 = exp(-sum(distance.^2,1)/(2*width^2));
distance = P - C(:,2);
F2 = exp(-sum(distance.^2,1)/(2*width^2));
F = [F1; F2];

% 2. Add a row of ones for the output bias.
A = [F; ones(1,size(F,2))];

% 3. Find linear output weights by least squares.
W = T*A'/(A*A' + 1e-4*eye(3));

% 4. Choose the class with the largest output.
scores = W*A;
[~, predicted] = max(scores, [], 1);
predicted = predicted - 1;    % MATLAB indices 1,2 -> classes 0,1

disp(table(labels', predicted', ...
    'VariableNames', {'Expected','Predicted'}));

% In Lab4, P contains 35 image features and T has 10 rows for digits 0...9.
% The idea is exactly the same; only the number of features and classes changes.
