%% Settings
clear
seed     = 0;
filters  = [4 8 16];                 % baseline = [8 16 32]
epochs = 20;
saveName = "daniel_cnn_optimized";

%% Load data
rng(seed);
load('../../data/standardized.mat');

XTrain = num2cell(trainStandardized', 1)';
XVal   = num2cell(valStandardized', 1)';
XTest  = num2cell(testStandardized', 1)';

YTrain = categorical(trainLabels(:));
YVal   = categorical(valLabels(:));
YTest  = categorical(testLabels(:));
classNames = categories(YTrain);

%% Build the CNN
layers = [
    sequenceInputLayer(1, MinLength=5000)

    convolution1dLayer(32, filters(1), Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(4, Stride=4)

    convolution1dLayer(16, filters(2), Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(4, Stride=4)

    convolution1dLayer(8, filters(3), Padding="same")
    batchNormalizationLayer
    reluLayer

    globalAveragePooling1dLayer
    fullyConnectedLayer(3)
    softmaxLayer
];

%% Train
options = trainingOptions("adam", ...
    MaxEpochs=epochs, ...
    MiniBatchSize=32, ...
    Shuffle="every-epoch", ...
    ValidationData={XVal, YVal}, ...
    Metrics="accuracy", ...
    Plots="training-progress", ...
    Verbose=false);

net = trainnet(XTrain, YTrain, layers, "crossentropy", options);

%% Results
YPred = scores2label(minibatchpredict(net, XTest), classNames);
testAccuracy = mean(YPred == YTest)
confusionchart(YTest, YPred);
metrics = estimateNetworkMetrics(net);
learnables = sum(metrics.NumberOfLearnables)
flashKB = sum(metrics.("ParameterMemory (MB)")) * 1024

%% Save
save("../../models/uncompressed/" + saveName + ".mat", 'net');