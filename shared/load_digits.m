function [P,T,Ptest,labels,testLabels] = load_digits(dataFolder)
% Write each row in this exact order: 0123456789.
trainRows = 2;
testRows = 1;
trainFile = fullfile(dataFolder,'train_digits.png');
testFile = fullfile(dataFolder,'test_digits.png');
assert(isfile(trainFile) && isfile(testFile), ...
    'Add train_digits.png and test_digits.png to data/. See data/INSTRUCTIONS.md.');
[P,trainTiles] = digit_features(trainFile,trainRows);
[Ptest,testTiles] = digit_features(testFile,testRows);
labels = repmat(0:9,1,trainRows);
testLabels = repmat(0:9,1,testRows);
assert(size(P,2) == numel(labels), 'Training image must have 10 digits per row.');
assert(size(Ptest,2) == numel(testLabels), 'Test image must have 10 digits per row.');
T = repmat(eye(10),1,trainRows); % One-hot targets: 10 outputs.
figure('Name','Check training segmentation');
for i = 1:numel(trainTiles)
    subplot(trainRows,10,i); imagesc(trainTiles{i}); axis image off;
    title(num2str(labels(i)));
end
colormap(gray);
figure('Name','Check test segmentation');
for i = 1:numel(testTiles)
    subplot(testRows,10,i); imagesc(testTiles{i}); axis image off;
    title(num2str(testLabels(i)));
end
colormap(gray);
end
