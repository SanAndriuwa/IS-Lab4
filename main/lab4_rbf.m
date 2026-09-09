% Main task: compare 13 and 8 RBF neurons on the SAME held-out digits.
clear; clc; close all;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'shared'));
[P,T,Ptest,labels,testLabels] = load_digits(fullfile(root,'data'));
counts = [13 8];
width = 1;
trainAccuracy = zeros(2,1); testAccuracy = zeros(2,1);
for j = 1:2
    model = train_rbf(P,T,counts(j),width);
    Y = model.W*[rbf_features(P,model.C,width);ones(1,size(P,2))];
    Ytest = model.W*[rbf_features(Ptest,model.C,width);ones(1,size(Ptest,2))];
    [~,pred] = max(Y,[],1); pred = pred-1;
    [~,predTest] = max(Ytest,[],1); predTest = predTest-1;
    trainAccuracy(j) = 100*mean(pred == labels);
    testAccuracy(j) = 100*mean(predTest == testLabels);
    fprintf('\nRBF neurons: %d\n',counts(j));
    disp(table(testLabels',predTest','VariableNames',{'Expected','Predicted'}));
end
disp(table(counts',trainAccuracy,testAccuracy, ...
    'VariableNames',{'Neurons','TrainingPercent','TestPercent'}));
figure; bar(counts,testAccuracy); ylim([0 100]); grid on;
xlabel('RBF neurons'); ylabel('Test accuracy (%)');
title('13 versus 8 RBF neurons');
