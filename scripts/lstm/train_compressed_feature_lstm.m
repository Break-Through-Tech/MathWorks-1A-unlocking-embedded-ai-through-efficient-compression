% Train smaller standard feature LSTMs and calibrate them for quantization.
%
% This experiment keeps the feature extraction, scaler, data splits, and
% training schedule of the saved 64-unit feature-LSTM baseline. It trains
% 48- and 32-unit standard LSTMs, selects a model using validation accuracy,
% evaluates both candidates on the test split, and creates calibrated
% dlquantizer objects. When Fixed-Point Designer is available, the companion
% calibration function also creates and evaluates simulatable INT8 networks.

clear;
clc;
rng(42);

repoRoot = fileparts(fileparts(fileparts(mfilename("fullpath"))));
modelsDir = fullfile(repoRoot, "models");
baselineFile = fullfile(modelsDir, "uncompressed", ...
    "alex_feature_lstm_64.mat");
if ~isfile(baselineFile)
    error("Run train_feature_lstm_fault_classifier.m before this experiment.");
end

baseline = load(baselineFile, "net", "metrics", "featureConfig", ...
    "featureNames", "featureMu", "featureSigma", "classNames");
trainSet = load(fullfile(repoRoot, "data", "train.mat"));
valSet = load(fullfile(repoRoot, "data", "val.mat"));

featuresTrain = extract_bearing_features(trainSet.trainData, baseline.featureConfig);
featuresVal = extract_bearing_features(valSet.valData, baseline.featureConfig);
XTrain = (featuresTrain - baseline.featureMu) ./ baseline.featureSigma;
XVal = (featuresVal - baseline.featureMu) ./ baseline.featureSigma;
YTrain = categorical(string(trainSet.trainLabels), baseline.classNames);
YVal = categorical(string(valSet.valLabels), baseline.classNames);
assert(~any(isundefined(YTrain)) && ~any(isundefined(YVal)), ...
    "Unexpected class label in the training or validation split.");

hiddenUnitCandidates = [48 32];
miniBatchSize = 32;
maxValidationAccuracyDrop = 1.0; % Percentage points relative to baseline.
numCandidates = numel(hiddenUnitCandidates);
candidateNets = cell(numCandidates, 1);
candidateTrainingInfo = cell(numCandidates, 1);
candidateModelFiles = strings(numCandidates, 1);

[baselineLearnables, baselineBytes] = networkLearnablesSize(baseline.net);
baselineValidationAccuracy = baseline.metrics.ValidationAccuracy;
baselineTestAccuracy = baseline.metrics.TestAccuracy;

hiddenUnits = [64; hiddenUnitCandidates(:)];
numLearnables = zeros(numCandidates + 1, 1);
learnablesBytes = zeros(numCandidates + 1, 1);
validationAccuracy = zeros(numCandidates + 1, 1);
testAccuracy = nan(numCandidates + 1, 1);
macroF1 = nan(numCandidates + 1, 1);
numLearnables(1) = baselineLearnables;
learnablesBytes(1) = baselineBytes;
validationAccuracy(1) = baselineValidationAccuracy;
testAccuracy(1) = baselineTestAccuracy;

for i = 1:numCandidates
    numHiddenUnits = hiddenUnitCandidates(i);
    layers = [
        sequenceInputLayer(numel(baseline.featureNames), Name="features")
        lstmLayer(numHiddenUnits, OutputMode="last", Name="lstm")
        dropoutLayer(0.20, Name="dropout")
        fullyConnectedLayer(numel(baseline.classNames), Name="fc")
        softmaxLayer(Name="softmax")];

    options = trainingOptions("adam", ...
        MaxEpochs=60, ...
        MiniBatchSize=miniBatchSize, ...
        InitialLearnRate=1e-3, ...
        LearnRateSchedule="piecewise", ...
        LearnRateDropPeriod=20, ...
        LearnRateDropFactor=0.5, ...
        Shuffle="every-epoch", ...
        ValidationData={XVal, YVal}, ...
        ValidationFrequency=max(1, floor(numel(YTrain)/miniBatchSize)), ...
        OutputNetwork="best-validation", ...
        Metrics="accuracy", ...
        ExecutionEnvironment="cpu", ...
        Plots="none", ...
        Verbose=true, ...
        VerboseFrequency=60);

    fprintf("Training the %d-unit feature LSTM.\n", numHiddenUnits);
    [net, trainingInfo] = trainnet(XTrain, YTrain, layers, ...
        "crossentropy", options);
    candidateNets{i} = net;
    candidateTrainingInfo{i} = trainingInfo;
    [numLearnables(i+1), learnablesBytes(i+1)] = networkLearnablesSize(net);

    validationScores = minibatchpredict(net, XVal, ...
        MiniBatchSize=miniBatchSize, ExecutionEnvironment="cpu");
    validationLabels = scores2label(validationScores, baseline.classNames);
    validationAccuracy(i+1) = 100*mean(validationLabels == YVal);

    candidateModelFiles(i) = fullfile(modelsDir, ...
        sprintf("lstm_feature_%d_fault_classifier.mat", numHiddenUnits));
    featureConfig = baseline.featureConfig;
    featureNames = baseline.featureNames;
    featureMu = baseline.featureMu;
    featureSigma = baseline.featureSigma;
    classNames = baseline.classNames;
    save(candidateModelFiles(i), "net", "trainingInfo", "layers", "options", ...
        "featureConfig", "featureNames", "featureMu", "featureSigma", ...
        "classNames", "numHiddenUnits");

    codegenModelFile = fullfile(modelsDir, ...
        sprintf("lstm_feature_%d_codegen_network.mat", numHiddenUnits));
    save(codegenModelFile, "net");

end

% Load the test split only after all training and validation-based selection.
testSet = load(fullfile(repoRoot, "data", "test.mat"));
featuresTest = extract_bearing_features(testSet.testData, baseline.featureConfig);
XTest = (featuresTest - baseline.featureMu) ./ baseline.featureSigma;
YTest = categorical(string(testSet.testLabels), baseline.classNames);
assert(~any(isundefined(YTest)), "Unexpected class label in the test split.");

baselineScores = minibatchpredict(baseline.net, XTest, ...
    MiniBatchSize=miniBatchSize, ExecutionEnvironment="cpu");
baselineLabels = scores2label(baselineScores, baseline.classNames);
baselineCounts = confusionmat(YTest, baselineLabels, ...
    Order=categorical(baseline.classNames, baseline.classNames));
macroF1(1) = calculateMacroF1(baselineCounts);

for i = 1:numCandidates
    scoresTest = minibatchpredict(candidateNets{i}, XTest, ...
        MiniBatchSize=miniBatchSize, ExecutionEnvironment="cpu");
    YPredTest = scores2label(scoresTest, baseline.classNames);
    testAccuracy(i+1) = 100*mean(YPredTest == YTest);
    confusionCounts = confusionmat(YTest, YPredTest, ...
        Order=categorical(baseline.classNames, baseline.classNames));
    macroF1(i+1) = calculateMacroF1(confusionCounts);

    % Add evaluation outputs to the previously saved artifact.
    modelFile = candidateModelFiles(i);
    save(modelFile, "scoresTest", "YPredTest", "confusionCounts", "-append");
end

compressionPercent = 100*(1 - numLearnables/baselineLearnables);
validationAccuracyDrop = baselineValidationAccuracy - validationAccuracy;
selected = false(numCandidates + 1, 1);
eligible = find(validationAccuracyDrop(2:end) <= maxValidationAccuracyDrop) + 1;
if isempty(eligible)
    [~, relativeIndex] = max(validationAccuracy(2:end));
    selected(relativeIndex + 1) = true;
else
    [~, relativeIndex] = max(compressionPercent(eligible));
    selected(eligible(relativeIndex)) = true;
end

model = ["Feature LSTM baseline"; ...
    compose("Feature LSTM %d-unit", hiddenUnitCandidates(:))];
compressionResults = table(model, hiddenUnits, numLearnables, learnablesBytes, ...
    learnablesBytes/1024, compressionPercent, validationAccuracy, ...
    validationAccuracyDrop, testAccuracy, macroF1, selected, ...
    VariableNames=["Model", "HiddenUnits", "NumLearnables", ...
    "LearnablesBytes", "LearnablesKiB", "CompressionPercent", ...
    "ValidationAccuracy", "ValidationAccuracyDrop", "TestAccuracy", ...
    "MacroF1", "Selected"]);

quantizationResults = calibrate_feature_lstm_int8;

resultsFile = fullfile(modelsDir, "lstm_feature_compression_results.mat");
save(resultsFile, "compressionResults", "quantizationResults", ...
    "maxValidationAccuracyDrop");
writetable(compressionResults, ...
    fullfile(modelsDir, "lstm_feature_compression_results.csv"));
plotCompressionResults(compressionResults, modelsDir);
disp(compressionResults);
disp(quantizationResults);
fprintf("Selected using validation accuracy: %s\n", ...
    compressionResults.Model(compressionResults.Selected));
fprintf("Saved compression results to %s\n", resultsFile);

function [numLearnables, totalBytes] = networkLearnablesSize(net)
values = net.Learnables.Value;
numLearnables = 0;
totalBytes = 0;
for k = 1:numel(values)
    value = extractdata(values{k});
    valueInfo = whos("value");
    numLearnables = numLearnables + numel(value);
    totalBytes = totalBytes + valueInfo.bytes;
end
end

function macroF1 = calculateMacroF1(counts)
truePositive = diag(counts);
falseNegative = sum(counts, 2) - truePositive;
falsePositive = sum(counts, 1)' - truePositive;
precision = truePositive ./ max(truePositive + falsePositive, 1);
recall = truePositive ./ max(truePositive + falseNegative, 1);
f1 = 2*precision.*recall ./ max(precision + recall, eps);
macroF1 = mean(f1);
end

function plotCompressionResults(results, modelsDir)
[compression, order] = sort(results.CompressionPercent);
validation = results.ValidationAccuracy(order);
test = results.TestAccuracy(order);
labels = compose("%d units", results.HiddenUnits(order));

fig = figure(Visible="off", Color="white", Position=[100 100 820 520]);
plot(compression, validation, "-o", LineWidth=1.8, MarkerSize=8);
hold on;
plot(compression, test, "--s", LineWidth=1.8, MarkerSize=8);
for k = 1:numel(compression)
    text(compression(k), validation(k), "  " + labels(k), ...
        VerticalAlignment="bottom");
end
xlabel("Learnable-parameter compression (%)");
ylabel("Accuracy (%)");
title("Feature LSTM Compression versus Accuracy");
legend("Validation", "Test", Location="best");
grid on;
exportgraphics(fig, fullfile(modelsDir, ...
    "lstm_feature_compression_accuracy.png"), Resolution=180);
close(fig);
end
