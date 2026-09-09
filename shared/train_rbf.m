function model = train_rbf(P,T,count,width)
% Choose diverse training examples as centers (farthest-point selection).
% Fit the linear output layer by regularized least squares.
assert(count <= size(P,2), 'Not enough training digits for this neuron count.');
indices = zeros(1,count); indices(1) = 1;
nearest = inf(1,size(P,2));
for k = 2:count
    dist = sum((P-P(:,indices(k-1))).^2,1);
    nearest = min(nearest,dist);
    nearest(indices(1:k-1)) = -inf;
    [~,indices(k)] = max(nearest);
end
model.C = P(:,indices);
model.width = width;
F = rbf_features(P,model.C,width);
A = [F;ones(1,size(P,2))];
lambda = 1e-4; % Small regularization for a stable linear solve.
model.W = (T*A')/(A*A'+lambda*eye(count+1));
end
