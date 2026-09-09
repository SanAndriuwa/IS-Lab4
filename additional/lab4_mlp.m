% Additional task: MLP using the same features and split as the RBF task.
clear; clc; close all;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'shared'));
[P,T,Ptest,labels,testLabels] = load_digits(fullfile(root,'data'));
[model,loss] = train_mlp(P,T,20,3000,0.1);
[~,pred] = max(predict_mlp(model,P),[],1); pred = pred-1;
[~,predTest] = max(predict_mlp(model,Ptest),[],1); predTest = predTest-1;
fprintf('MLP training accuracy: %.1f %%\n',100*mean(pred == labels));
fprintf('MLP test accuracy: %.1f %%\n',100*mean(predTest == testLabels));
disp(table(testLabels',predTest','VariableNames',{'Expected','Predicted'}));
figure; semilogy(loss); grid on;
xlabel('Epoch'); ylabel('Cross-entropy'); title('MLP training');
