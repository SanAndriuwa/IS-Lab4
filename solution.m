clear;
clc;
close all;

%% Lab4: reuse the original image processing and recognition workflow.
% Required: Image Processing Toolbox and Deep Learning Toolbox
% (or older Neural Network Toolbox with newrb and feedforwardnet).
%
% Place these RGB PNG images next to this script:
% train_digits.png: TWO rows, each containing 0123456789.
% test_digits.png: ONE separately handwritten row containing 0123456789.
% Use dark ink, white paper, straight rows and generous gaps between digits.
% The original extractor assumes the same number of symbols in every row.
% Its 7-by-7 dilation can merge nearby digits; inspect the displayed crops.
%
% This script calls the unchanged pozymiai_raidems_atpazinti.m directly.
% RBF training and recognition follow vaizdo_atpazinimas.m.
% Unlike its P(:,12:22) check, our test image is not used for training.
% The MLP additional task follows the RBF comparison below.
folder = fileparts(mfilename('fullpath'));
addpath(folder);
dataFolder = folder;

%% Settings: change row counts here if you add complete rows.
trainRows = 2;
testRows = 1;
trainFile = fullfile(dataFolder, 'train_digits.png');
testFile = fullfile(dataFolder, 'test_digits.png');
assert(isfile(trainFile) && isfile(testFile), ...
    'Place train_digits.png and test_digits.png next to solution.m.');

% The original function calls rgb2gray unconditionally: use RGB PNG files.
trainImage = imread(trainFile);
testImage = imread(testFile);
assert(ndims(trainImage) == 3 && size(trainImage,3) == 3, ...
    'Save train_digits.png as an RGB image for the original extractor.');
assert(ndims(testImage) == 3 && size(testImage,3) == 3, ...
    'Save test_digits.png as an RGB image for the original extractor.');

%% Original calls: image -> cell array of 35-element feature columns.
pozymiai_tinklo_mokymui = pozymiai_raidems_atpazinti(trainFile, trainRows);
P = cell2mat(pozymiai_tinklo_mokymui);

% Inspect training crops before the original function reuses its figures.
disp('Check training crops in figures 5 and 6: each row must be 0123456789.');
disp('Press any key in MATLAB to continue to the test image.');
pause;

pozymiai_patikrai = pozymiai_raidems_atpazinti(testFile, testRows);
P2 = cell2mat(pozymiai_patikrai);

%% Targets: one output per digit, repeated for every training row.
labels = repmat(0:9, 1, trainRows);
testLabels = repmat(0:9, 1, testRows);
T = repmat(eye(10), 1, trainRows);
assert(isequal(size(P), [35 numel(labels)]), ...
    'Expected 10 training digits per row and 35 features per digit.');
assert(isequal(size(P2), [35 numel(testLabels)]), ...
    'Expected 10 test digits per row and 35 features per digit.');
assert(all(isfinite(P(:))) && all(isfinite(P2(:))), ...
    'Image features contain invalid values. Check the segmented images.');
disp('Check test crops in figures 5 and 6: each row must be 0123456789.');
disp('Press any key in MATLAB to continue to network training.');
pause;

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

