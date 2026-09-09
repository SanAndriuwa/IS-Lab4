function model = train_rbf(P, T, count, width)
% Choose diverse training examples as RBF centers.
% Then fit the linear output layer by regularized least squares.

assert(count <= size(P,2), ...
    'Not enough training digits for this neuron count.');

% Farthest-point selection: each new center is far from the old centers.
indices = zeros(1, count);
indices(1) = 1;
nearest = inf(1, size(P,2));
for k = 2:count
    distance = sum((P - P(:, indices(k-1))).^2, 1);
    nearest = min(nearest, distance);
    nearest(indices(1:k-1)) = -inf;
    [~, indices(k)] = max(nearest);
end

model.C = P(:, indices);
model.width = width;

% F contains one row per RBF neuron and one column per digit.
F = rbf_features(P, model.C, width);
% Add a row of ones so the linear model has a bias term.
A = [F; ones(1, size(P,2))];

% Solve W*A ≈ T. Lambda makes the matrix inversion more stable.
lambda = 1e-4;
G = A*A' + lambda*eye(count + 1);
model.W = (T*A') / G;
end
