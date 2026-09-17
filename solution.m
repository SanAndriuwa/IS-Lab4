clear;
clc;
close all;

folder = fileparts(mfilename('fullpath'));
addpath(folder);

%% 1. Read the drawing using the original feature extractor.
filename = fullfile(folder, 'digits.png');
rows = 5;
assert(isfile(filename), 'Save your five-row Paint drawing as digits.png.');
drawing = imread(filename);
% The original feature extractor calls rgb2gray, so use an RGB image.
assert(ndims(drawing) == 3 && size(drawing,3) == 3, ...
    'Save digits.png as an RGB/color PNG.');

pozymiai = pozymiai_raidems_atpazinti(filename, rows);
P_all = cell2mat(pozymiai); % 35 rows of features, 50 columns of digits.
assert(isequal(size(P_all), [35 50]), ...
    'Expected 50 digits: five rows of ten digits. Check gaps and crops.');
assert(all(isfinite(P_all(:))), 'Invalid features: check the image crops.');

% Check the segmentation before training.
disp('Check figures 5 and 6: rows 1-4 are 0123456789; row 5 is 5612907834.');
disp('Each tile must contain one complete digit. Stop if the crops are wrong.');
disp('Press any key in MATLAB to continue.');
pause;

%% 2. Split data into training and testing sets.
% Columns 1...40 = first four rows for training.
% Columns 41...50 = fifth row for testing.
P = P_all(:, 1:40);
P2 = P_all(:, 41:50);

% Ten outputs represent digits 0...9 in one-hot form.
T = repmat(eye(10), 1, 4);
testLabels = [5 6 1 2 9 0 7 8 3 4];

%% 3. Main task: compare RBF networks with limits 13 and 8.
counts = [13 8];
spread = 1;
accuracy = zeros(1, length(counts));
actualCounts = zeros(1, length(counts));

for n = 1:length(counts)
    % Directly reuse vaizdo_atpazinimas.m.
    [predicted, atsakymas, ~, actualCounts(n)] = ...
        vaizdo_atpazinimas(P, T, P2, counts(n), spread);

    disp(atsakymas);
    disp('Rows: expected digits, predicted digits');
    disp([testLabels; predicted]);

    accuracy(n) = 100 * mean(predicted == testLabels);
    fprintf('RBF limit: %d; actual neurons: %d; test accuracy: %.1f%%\n', ...
        counts(n), actualCounts(n), accuracy(n));
end

%% 4. Compare the two RBF results.
disp(table(counts', actualCounts', accuracy', ...
    'VariableNames', {'NeuronLimit', 'ActualNeurons', 'TestPercent'}));

figure;
bar(counts, accuracy);
ylim([0 100]);
xlabel('Maximum RBF neurons');
ylabel('Test accuracy (%)');
title('RBF comparison: 13 versus 8');

%% 5. Additional task: replace RBF with a multilayer perceptron.
% Network structure: 35 input features -> 20 hidden neurons -> 10 outputs.
rng(4);
hidden = 20;
tinklas = feedforwardnet(hidden, 'trainscg');
tinklas.layers{1}.transferFcn = 'tansig';
tinklas.layers{2}.transferFcn = 'purelin';
tinklas.performFcn = 'mse';

% Use all 40 P columns for training. P2 stays separate for final testing.
tinklas.divideFcn = 'dividetrain';
tinklas.trainParam.epochs = 1000;
tinklas.trainParam.showWindow = false;
tinklas = train(tinklas, P, T);

%% 6. Test the MLP using the same recognition idea: sim -> max.
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
