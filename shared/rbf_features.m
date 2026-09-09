function F = rbf_features(P,C,width)
% P: 35 x N; C: 35 x K; output F: K x N.
F = zeros(size(C,2),size(P,2));
for k = 1:size(C,2)
    distance2 = sum((P-C(:,k)).^2,1);
    F(k,:) = exp(-distance2/(2*width^2));
end
end
