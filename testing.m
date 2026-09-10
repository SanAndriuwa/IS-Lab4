clear;
clc;
close all;

%% Load 35 image features and targets for digits 0 to 9
% Add your own images as described in data/INSTRUCTIONS.md.
folder = fileparts(mfilename('fullpath'));
addpath(fullfile(folder, 'shared'));
[x, target, x_naujas, labels, testLabels] = load_digits(fullfile(folder, 'data'));
epoch = 2000;
eta = 0.05;
r = 1;
counts = [13 8];
accuracy = zeros(1,2);

%% Compare two networks using the same training and test images
for n = 1:length(counts)
    hidden = counts(n);

    % Take evenly spaced training examples as fixed centers.
    indices = round(linspace(1, size(x,2), hidden));
    c = x(:,indices);

    % One output per digit; w(k,h) connects hidden h to output k.
    w = 0.1*randn(10,hidden);
    b = zeros(10,1);
    y_hidden = zeros(hidden,1);
    y_output = zeros(10,1);
    delta = zeros(10,1);

    %% Training
    for j = 1:epoch
        for i = 1:size(x,2)
            % Gaussian hidden neurons
            for h = 1:hidden
                distance = sum((x(:,i)-c(:,h)).^2);
                y_hidden(h) = exp(-distance/(2*r^2));
            end

            % Linear output neurons and their deltas
            for k = 1:10
                v = b(k);
                for h = 1:hidden
                    v = v + y_hidden(h)*w(k,h);
                end
                y_output(k) = v;
                e = target(k,i) - y_output(k);
                delta(k) = e;
            end

            % Update output weights and biases
            for k = 1:10
                for h = 1:hidden
                    w(k,h) = w(k,h) + eta*delta(k)*y_hidden(h);
                end
                b(k) = b(k) + eta*delta(k);
            end
        end
    end

    %% Testing: no weight updates
    Y = zeros(size(testLabels));
    for i = 1:size(x_naujas,2)
        for h = 1:hidden
            distance = sum((x_naujas(:,i)-c(:,h)).^2);
            y_hidden(h) = exp(-distance/(2*r^2));
        end
        for k = 1:10
            v = b(k);
            for h = 1:hidden
                v = v + y_hidden(h)*w(k,h);
            end
            y_output(k) = v;
        end
        [~, index] = max(y_output);
        Y(i) = index - 1; % Output 1 represents digit 0.
    end

    accuracy(n) = 100*mean(Y == testLabels);
    fprintf('Hidden neurons: %d; test accuracy: %.1f%%\n', hidden, accuracy(n));
    disp('Rows: target digits, predicted digits');
    disp([testLabels; Y]);

    figure;
    plot(1:length(Y), testLabels, 'o');
    hold on;
    plot(1:length(Y), Y, '*');
    xlabel('Test example');
    ylabel('Digit');
    legend('Target', 'Network');
    title(sprintf('Testing: %d RBF neurons', hidden));
end

figure;
bar(counts, accuracy);
ylim([0 100]);
xlabel('RBF neurons');
ylabel('Test accuracy (%)');
title('13 versus 8 hidden neurons');
