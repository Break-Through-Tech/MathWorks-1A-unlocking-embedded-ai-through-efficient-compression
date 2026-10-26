%% Settings
clear
seed     = 0;
f1       = 8;     % conv filters
nLSTM    = 16;    % LSTM units
nGRU     = 16;    % GRU units
epochs   = 20;
saveName = "daniel_hybrid_lstm_gru";

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

%% Build the hybrid
layers = [
    sequenceInputLayer(1, MinLength=5000)

    convolution1dLayer(32, f1, Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(16, Stride=16)        % shrink 5000 -> ~312 steps

    lstmLayer(nLSTM)
    gruLayer(nGRU, OutputMode="last")

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
    GradientThreshold=1, ...
    Plots="training-progress", ...
    Verbose=false);

net = trainnet(XTrain, YTrain, layers, "crossentropy", options);

%% Results
YPred = scores2label(minibatchpredict(net, XTest), classNames);
testAccuracy = mean(YPred == YTest)
learnables = sum(cellfun(@numel, net.Learnables.Value))
modelSizeKB = learnables * 4 / 1024

%% Confusion matrix
figure;
cm = confusionchart(YTest, YPred);
cm.Title = strrep(saveName, "_", " ") + " (test set)";

%% Save
save("../../models/uncompressed/" + saveName + ".mat", 'net');