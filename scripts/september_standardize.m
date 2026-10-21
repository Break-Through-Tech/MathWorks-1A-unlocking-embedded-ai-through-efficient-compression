% Standardization - all classes together, using training statistics only
clear
load('../data/train.mat'); load('../data/val.mat'); load('../data/test.mat');

% Compute mean and std from TRAINING data only
mu = mean(trainData(:));
sigma = std(trainData(:));

% Convert all three sets using the same two numbers
trainStandardized = (trainData - mu) / sigma;
valStandardized   = (valData   - mu) / sigma;
testStandardized  = (testData  - mu) / sigma;

% Check 1: range before and after
fprintf('Before: min=%.2f  max=%.2f\n', min(trainData(:)), max(trainData(:)));
fprintf('After:  min=%.2f  max=%.2f\n', min(trainStandardized(:)), max(trainStandardized(:)));

% Check 2: train should be ~0 and ~1; val and test close but not exact
fprintf('Train: mean=%.3f std=%.3f\n', mean(trainStandardized(:)), std(trainStandardized(:)));
fprintf('Val:   mean=%.3f std=%.3f\n', mean(valStandardized(:)),   std(valStandardized(:)));
fprintf('Test:  mean=%.3f std=%.3f\n', mean(testStandardized(:)),  std(testStandardized(:)));

% Check 3: class differences should be preserved (Inner ~2.4x Normal)
r = rms(trainStandardized, 2);
classes = unique(trainLabels);
for k = 1:3
    fprintf('%s RMS: %.3f\n', classes{k}, mean(r(strcmp(trainLabels, classes{k}))));
end

% Uncomment to save standardized data. A copy is already in data/standardized.mat
% save('../data/standardized.mat', 'trainStandardized', 'valStandardized', 'testStandardized', ...
%     'trainLabels', 'valLabels', 'testLabels', 'mu', 'sigma');



