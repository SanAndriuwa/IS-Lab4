function [model,loss] = train_mlp(P,T,hidden,epochs,eta)
% Manual full-batch backpropagation: tanh hidden layer, softmax outputs.
rng(4);
n = size(P,2);
model.W1 = 0.1*randn(hidden,size(P,1));
model.b1 = zeros(hidden,1);
model.W2 = 0.1*randn(size(T,1),hidden);
model.b2 = zeros(size(T,1),1);
loss = zeros(1,epochs);
for epoch = 1:epochs
    H = tanh(model.W1*P+model.b1);
    Z = model.W2*H+model.b2;
    E = exp(Z-max(Z,[],1));
    Y = E./sum(E,1);
    loss(epoch) = -sum(sum(T.*log(max(Y,1e-12))))/n;
    D2 = (Y-T)/n;
    D1 = (model.W2'*D2).*(1-H.^2);
    % D1 must be computed before W2 is changed.
    model.W2 = model.W2-eta*(D2*H');
    model.b2 = model.b2-eta*sum(D2,2);
    model.W1 = model.W1-eta*(D1*P');
    model.b1 = model.b1-eta*sum(D1,2);
end
end
