clear;
clc;
close all;

%% Load the same images as in testing.m
folder = fileparts(mfilename('fullpath'));
addpath(fullfile(folder, 'shared'));
[x, target, x_naujas, labels, testLabels] = load_digits(fullfile(folder, 'data'));
epoch = 3000;
eta = 0.01;
hidden = 20;

%% Weights and biases: 35 inputs, 20 hidden neurons, 10 outputs
w1 = 0.1*randn(hidden,35);
b1 = zeros(hidden,1);
w2 = 0.1*randn(10,hidden);
b2 = zeros(10,1);
y_hidden = zeros(hidden,1);
y_output = zeros(10,1);
delta_output = zeros(10,1);
delta_hidden = zeros(hidden,1);

%% Training
for j = 1:epoch
    for i = 1:size(x,2)
        % Hidden layer: tanh, just like Lab2.
        for h = 1:hidden
            v = b1(h);
            for p = 1:35
                v = v + x(p,i)*w1(h,p);
            end
            y_hidden(h) = tanh(v);
        end

        % Linear outputs: one score for each digit.
        for k = 1:10
            v = b2(k);
            for h = 1:hidden
                v = v + y_hidden(h)*w2(k,h);
            end
            y_output(k) = v;
            e = target(k,i) - y_output(k);
            delta_output(k) = e;
        end

        % Each hidden neuron receives errors from ALL ten outputs.
        % Calculate all deltas before changing any weights.
        for h = 1:hidden
            error_sum = 0;
            for k = 1:10
                error_sum = error_sum + delta_output(k)*w2(k,h);
            end
            delta_hidden(h) = (1-y_hidden(h)^2)*error_sum;
        end

        % Update output weights and biases.
        for k = 1:10
            for h = 1:hidden
                w2(k,h) = w2(k,h) + eta*delta_output(k)*y_hidden(h);
            end
            b2(k) = b2(k) + eta*delta_output(k);
        end

        % Update hidden weights and biases.
        for h = 1:hidden
            for p = 1:35
                w1(h,p) = w1(h,p) + eta*delta_hidden(h)*x(p,i);
            end
            b1(h) = b1(h) + eta*delta_hidden(h);
        end
    end
end

%% Testing: no weight updates
Y = zeros(size(testLabels));
for i = 1:size(x_naujas,2)
    for h = 1:hidden
        v = b1(h);
        for p = 1:35
            v = v + x_naujas(p,i)*w1(h,p);
        end
        y_hidden(h) = tanh(v);
    end
    for k = 1:10
        v = b2(k);
        for h = 1:hidden
            v = v + y_hidden(h)*w2(k,h);
        end
        y_output(k) = v;
    end
    [~, index] = max(y_output);
    Y(i) = index - 1; % Scores are not probabilities.
end
fprintf('Test accuracy: %.1f%%\n', 100*mean(Y == testLabels));
disp('Rows: target digits, predicted digits');
disp([testLabels; Y]);
figure;
plot(1:length(Y), testLabels, 'o');
hold on;
plot(1:length(Y), Y, '*');
legend('Target', 'MLP');
xlabel('Test example');
ylabel('Digit');
title('Testing');
