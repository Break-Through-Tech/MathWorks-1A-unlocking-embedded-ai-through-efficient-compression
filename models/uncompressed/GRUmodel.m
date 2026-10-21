clear
clc

load("standardizedBearingData.mat")

% Convert labels
trainLabels = categorical(trainLabels);
valLabels = categorical(valLabels);
testLabels = categorical(testLabels);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 1: Convert data into sequences
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This step converts each row of the standardized
% data into a time-series sequence, 
% so the GRU can process it step-by-step.

numTrain = size(trainDataStandardized,1);

XTrain = cell(numTrain,1);

for i = 1:numTrain
    XTrain{i} = trainDataStandardized(i,:);
end

numVal = size(valDataStandardized,1);

XVal = cell(numVal,1);

for i = 1:numVal
    XVal{i} = valDataStandardized(i,:);
end

numTest = size(testDataStandardized,1);

XTest = cell(numTest,1);

for i = 1:numTest
    XTest{i} = testDataStandardized(i,:);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 2: DEFINE GRU MODEL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the GRU architecture used to learn
% patterns in the vibration signals.

numClasses = numel(categories(trainLabels));

layers = [
    sequenceInputLayer(1)

    gruLayer(100,"OutputMode","last")
    % The GRU processes the vibration
    % signal sequentially and learns
    % which parts of the signal are 
    % important for identifying faults.

    fullyConnectedLayer(numClasses)

    softmaxLayer

    classificationLayer
];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 3: TRAINING OPTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Configure how the GRU model will be 
% trained and validated.

options = trainingOptions("adam", ...
    "MaxEpochs",80, ...
    "MiniBatchSize",32, ...
    "ValidationData",{XVal,valLabels}, ...
    "Shuffle","every-epoch", ...
    "Plots","training-progress", ...
    "Verbose",false);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 4: TRAIN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train the GRU model using the sequence
% data and training labels.

trainedNetwork = trainNetwork( ...
    XTrain, ...
    trainLabels, ...
    layers, ...
    options);