%% Part 1: Load data and put each recording in its own package
clear
rng(0);                                   % same results every run
load('../data/standardized.mat');

XTrain = num2cell(trainStandardized', 1)';
XVal   = num2cell(valStandardized', 1)';
XTest  = num2cell(testStandardized', 1)';

%% Part 2: Convert labels to categories
YTrain = categorical(trainLabels(:));
YVal   = categorical(valLabels(:));
YTest  = categorical(testLabels(:));

classNames = categories(YTrain);

%% Part 3: Build the 1D CNN
layers = [
    sequenceInputLayer(1, MinLength=5000)

    convolution1dLayer(32, 8, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(4, Stride=4)

    convolution1dLayer(16, 16, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(4, Stride=4)

    convolution1dLayer(8, 32, Padding="same")
    batchNormalizationLayer
    reluLayer

    globalAveragePooling1dLayer
    fullyConnectedLayer(3)
    softmaxLayer
    ];

%% Part 4: Training settings
options = trainingOptions("adam", ...
    MaxEpochs=30, ...
    MiniBatchSize=32, ...
    Shuffle="every-epoch", ...
    ValidationData={XVal, YVal}, ...
    Metrics="accuracy", ...
    Plots="training-progress", ...
    Verbose=false);

%% Part 5: Train
net = trainnet(XTrain, YTrain, layers, "crossentropy", options);

%% Part 6: Test accuracy
scores = minibatchpredict(net, XTest);
YPred  = scores2label(scores, classNames);
testAccuracy = mean(YPred == YTest)
confusionchart(YTest, YPred);

%% Part 7: Flash estimate (weights)
metrics = estimateNetworkMetrics(net);
totalLearnables = sum(metrics.NumberOfLearnables)
flashMB = sum(metrics.("ParameterMemory (MB)"));
flashKB = flashMB * 1024

