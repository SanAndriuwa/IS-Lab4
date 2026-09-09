function Y = predict_mlp(model,P)
H = tanh(model.W1*P+model.b1);
Z = model.W2*H+model.b2;
E = exp(Z-max(Z,[],1));
Y = E./sum(E,1);
end
