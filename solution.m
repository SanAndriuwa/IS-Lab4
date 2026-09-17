clear;
clc;
close all;

folder = fileparts(mfilename('fullpath'));
addpath(folder);

%% 1. Read the drawing using the ORIGINAL feature extractor.
filename = fullfile(folder, 'digits.png');
rows = 5;
assert(isfile(filename), 'Save your five-row Paint drawing as digits.png.');
drawing = imread(filename);
% The unchanged original function calls rgb2gray, which needs an RGB image.
assert(ndims(drawing) == 3 && size(drawing,3) == 3, ...
    'Save digits.png as an RGB/color PNG.');

pozymiai = pozymiai_raidems_atpazinti(filename, rows);
P_all = cell2mat(pozymiai); % 35 rows of features, 50 columns of digits.
assert(isequal(size(P_all), [35 50]), ...
    'Expected 50 digits: five rows of ten digits. Check gaps and crops.');
assert(all(isfinite(P_all(:))), 'Invalid features: check the image crops.');

% The original segmentation assumes equally sized rows. Its dilation can
% merge nearby digits; disconnected strokes may produce extra objects.
% Incorrect counts may also cause an error inside the original function.
disp('Check figures 5 and 6: rows 1-4 are 0123456789; row 5 is 5612907834.');
disp('Each tile must contain one complete digit. Stop if the crops are wrong.');
disp('Press any key in MATLAB to continue.');
pause;

%% 2. First four rows train the networks; the fifth row tests them.
% Columns 1...10 = row 1, columns 11...20 = row 2, and so on.
P = P_all(:, 1:40);
P2 = P_all(:, 41:50);
% eye(10): output 1 means digit 0, output 2 means digit 1, etc.
T = repmat(eye(10), 1, 4);
testLabels = [5 6 1 2 9 0 7 8 3 4]; % Expected digits in the fifth row.
% P2 is never passed to newrb or train. It is used only for prediction.
% Ten test digits are a small test: one mistake changes accuracy by 10%.

%% Compare 13 and 8 RBF neurons using the same images.
counts = [13 8];
spread = 1;
accuracy = zeros(1, length(counts));
actualCounts = zeros(1, length(counts));

for n = 1:length(counts)
    % Original training call: goal = 0, spread = 1.
    % The last argument is the MAXIMUM number of hidden neurons.
    tinklas = newrb(P, T, 0, spread, counts(n));
    actualCounts(n) = tinklas.layers{1}.size;

    %% Recognize a separate test image, as in the original script.
    Y2 = sim(tinklas, P2);
    [~, b2] = max(Y2, [], 1);
    predicted = b2 - 1; % Outputs 1...10 represent digits 0...9.

    % Convert the output indices to readable digits.
    symbols = '0123456789';
    atsakymas = symbols(b2);
    disp(atsakymas);
    disp('Rows: expected digits, predicted digits');
    disp([testLabels; predicted]);

    accuracy(n) = 100 * mean(predicted == testLabels);
    fprintf('RBF limit: %d; actual neurons: %d; test accuracy: %.1f%%\n', ...
        counts(n), actualCounts(n), accuracy(n));
end

%% Compare the two results.
disp(table(counts', actualCounts', accuracy', ...
    'VariableNames', {'NeuronLimit', 'ActualNeurons', 'TestPercent'}));
figure;
bar(counts, accuracy);
ylim([0 100]);
xlabel('Maximum RBF neurons');
ylabel('Test accuracy (%)');
title('Original newrb workflow: 13 versus 8');


%% Additional task: replace newrb with a multilayer perceptron.
% 35 image features -> 20 hidden neurons -> 10 output scores.
rng(4);
hidden = 20;
tinklas = feedforwardnet(hidden, 'trainscg');
tinklas.layers{1}.transferFcn = 'tansig';
tinklas.layers{2}.transferFcn = 'purelin';
tinklas.performFcn = 'mse';

% Use all P columns for training. P2 is held out and never passed to train.
tinklas.divideFcn = 'dividetrain';
tinklas.trainParam.epochs = 1000;
tinklas.trainParam.showWindow = false;
tinklas = train(tinklas, P, T);

%% The prediction steps are the same as in the original script.
Y2 = sim(tinklas, P2);
[~, b2] = max(Y2, [], 1);
predicted = b2 - 1;
symbols = '0123456789';
atsakymas = symbols(b2);
disp(atsakymas);
disp('Rows: expected digits, predicted digits');
disp([testLabels; predicted]);
fprintf('MLP test accuracy: %.1f%%\n', 100 * mean(predicted == testLabels));

figure;
plot(1:length(testLabels), testLabels, 'o');
hold on;
plot(1:length(predicted), predicted, '*');
xlabel('Test example');
ylabel('Digit');
legend('Expected', 'MLP');
title('MLP with original image features');

