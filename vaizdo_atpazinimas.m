function [predicted, atsakymas, tinklas, actualNeurons] = vaizdo_atpazinimas(P, T, P2, neuronLimit, spread)
%VAIZDO_ATPAZINIMAS Train an RBF recognizer and classify test symbols.
%
% Inputs:
%   P           - training features. One column = one training symbol.
%   T           - correct training targets in one-hot form.
%   P2          - test features. One column = one unknown symbol.
%   neuronLimit - maximum number of hidden RBF neurons.
%   spread      - RBF spread parameter.
%
% Outputs:
%   predicted     - recognized digits as numbers 0...9.
%   atsakymas     - recognized digits as a character string.
%   tinklas       - trained RBF neural network.
%   actualNeurons - actual number of hidden RBF neurons created by newrb.
%
% The function keeps the same main recognition logic as the original
% laboratory example: newrb -> sim -> max.

%% 1. Create and train the RBF network.
% goal = 0 means newrb tries to reduce the training error as much as it can.
% neuronLimit is the MAXIMUM number of hidden neurons, not always the final one.
tinklas = newrb(P, T, 0, spread, neuronLimit);
actualNeurons = tinklas.layers{1}.size;

%% 2. Run the trained network on unknown/test symbols.
% Y2 has one column per test symbol and one row per output class.
Y2 = sim(tinklas, P2);

%% 3. Choose the output neuron with the largest value.
% b2 contains MATLAB indices 1...10.
[~, b2] = max(Y2, [], 1);

%% 4. Convert MATLAB output indices to digit labels 0...9.
predicted = b2 - 1;

% Output 1 = digit 0, output 2 = digit 1, ..., output 10 = digit 9.
symbols = '0123456789';
atsakymas = symbols(b2);
end
