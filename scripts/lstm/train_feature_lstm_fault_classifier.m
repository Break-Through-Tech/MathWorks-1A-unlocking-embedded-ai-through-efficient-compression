% Compare an LSTM on short feature sequences with the saved raw-signal LSTM.
% The architecture and schedule match the baseline; input representation changes.
% Parameters are fixed before evaluation. Test data is loaded after training.

clear;
clc;
rng(42);
repoRoot = fileparts(fileparts(fileparts(mfilename("fullpath"))));
modelsDir = fullfile(repoRoot, "models");
if ~isfolder(modelsDir)
    mkdir(modelsDir);
end
uncompressedModelsDir = fullfile(modelsDir, "uncompressed");
if ~isfolder(uncompressedModelsDir)
    mkdir(uncompressedModelsDir);
end

trainSet = load(fullfile(repoRoot, "data", "train.mat"));
valSet = load(fullfile(repoRoot, "data", "val.mat"));
classNames = ["InnerRaceFault", "Normal", "OuterRaceFault"];
featureNames = ["RMS", "Kurtosis", "CrestFactor", "HighBandPowerFraction"];
featureConfig = struct('signalLength', 5000, 'windowLength', 500, ...
    'hopLength', 250, 'sampleRate', 48828, 'bandStartHz', 48828/4);

featuresTrain = extract_bearing_features(trainSet.trainData, featureConfig);
featuresVal = extract_bearing_features(valSet.valData, featureConfig);
trainingFrames = reshape(permute(featuresTrain, [1 3 2]), [], numel(featureNames));
featureMu = single(mean(double(trainingFrames), 1));
featureSigma = single(std(double(trainingFrames), 0, 1));
featureSigma(featureSigma == 0) = 1;
XTrain = (featuresTrain - featureMu) ./ featureSigma;
XVal = (featuresVal - featureMu) ./ featureSigma;
YTrain = categorical(string(trainSet.trainLabels), classNames);
YVal = categorical(string(valSet.valLabels), classNames);
assert(~any(isundefined(YTrain)) && ~any(isundefined(YVal)), 'Unexpected class label.');

numHiddenUnits = 64;
miniBatchSize = 32;
layers = [
    sequenceInputLayer(numel(featureNames), Name="features")
    lstmLayer(numHiddenUnits, OutputMode="last", Name="lstm")
    dropoutLayer(0.20, Name="dropout")
    fullyConnectedLayer(numel(classNames), Name="fc")
    softmaxLayer(Name="softmax")];
options = trainingOptions("adam", ...
    MaxEpochs=60, MiniBatchSize=miniBatchSize, InitialLearnRate=1e-3, ...
    LearnRateSchedule="piecewise", LearnRateDropPeriod=20, LearnRateDropFactor=0.5, ...
    Shuffle="every-epoch", ValidationData={XVal, YVal}, ...
    ValidationFrequency=max(1, floor(numel(YTrain)/miniBatchSize)), ...
    OutputNetwork="best-validation", Metrics="accuracy", ...
    ExecutionEnvironment="cpu", Plots="training-progress", Verbose=true, VerboseFrequency=60);

fprintf("Input: %d frames x %d features per recording.\n", size(XTrain,1), size(XTrain,2));
trainingTimer = tic;
[net, trainingInfo] = trainnet(XTrain, YTrain, layers, "crossentropy", options);
trainingSeconds = toc(trainingTimer);

testSet = load(fullfile(repoRoot, "data", "test.mat"));
featuresTest = extract_bearing_features(testSet.testData, featureConfig);
XTest = (featuresTest - featureMu) ./ featureSigma;
YTest = categorical(string(testSet.testLabels), classNames);
assert(~any(isundefined(YTest)), 'Unexpected test class label.');
scoresTrain = minibatchpredict(net, XTrain, MiniBatchSize=miniBatchSize, ExecutionEnvironment="cpu");
scoresVal = minibatchpredict(net, XVal, MiniBatchSize=miniBatchSize, ExecutionEnvironment="cpu");
scoresTest = minibatchpredict(net, XTest, MiniBatchSize=miniBatchSize, ExecutionEnvironment="cpu");
YPredTrain = scores2label(scoresTrain, classNames);
YPredVal = scores2label(scoresVal, classNames);
YPredTest = scores2label(scoresTest, classNames);
classOrder = categorical(classNames, classNames);
confusionTrain = confusionmat(YTrain, YPredTrain, Order=classOrder);
confusionVal = confusionmat(YVal, YPredVal, Order=classOrder);
confusionCounts = confusionmat(YTest, YPredTest, Order=classOrder);
classMetrics = classificationMetrics(confusionCounts, classNames);

numLearnables = 0;
learnablesBytes = 0;
for k = 1:height(net.Learnables)
    value = extractdata(net.Learnables.Value{k});
    valueInfo = whos("value");
    numLearnables = numLearnables + numel(value);
    learnablesBytes = learnablesBytes + valueInfo.bytes;
end
metrics = table(100*mean(YPredTrain==YTrain), 100*mean(YPredVal==YVal), ...
    100*mean(YPredTest==YTest), numLearnables, learnablesBytes, ...
    learnablesBytes/1024, trainingSeconds, ...
    VariableNames=["TrainAccuracy", "ValidationAccuracy", "TestAccuracy", ...
    "NumLearnables", "LearnablesBytes", "LearnablesKiB", "TrainingSeconds"]);

comparisonMetrics = table("Feature LSTM", metrics.TrainAccuracy, ...
    metrics.ValidationAccuracy, metrics.TestAccuracy, classMetrics.Recall(3)*100, ...
    mean(classMetrics.F1Score), numLearnables, ...
    VariableNames=["Model", "TrainAccuracy", "ValidationAccuracy", "TestAccuracy", ...
    "OuterRaceRecall", "MacroF1", "NumLearnables"]);
baselineFile = fullfile(modelsDir, "lstm_fault_classifier.mat");
if isfile(baselineFile)
    baseline = load(baselineFile, "metrics", "classNames", "confusionCounts");
    assert(isequal(string(baseline.classNames(:)), classNames(:)), ...
        'Baseline class order does not match.');
    baselineClasses = classificationMetrics(baseline.confusionCounts, classNames);
    baselineMetrics = table("Raw-signal LSTM", baseline.metrics.TrainAccuracy, ...
        baseline.metrics.ValidationAccuracy, baseline.metrics.TestAccuracy, ...
        baselineClasses.Recall(3)*100, mean(baselineClasses.F1Score), ...
        baseline.metrics.NumLearnables, VariableNames=comparisonMetrics.Properties.VariableNames);
    comparisonMetrics = [baselineMetrics; comparisonMetrics];
end

modelFile = fullfile(uncompressedModelsDir, "alex_feature_lstm_64.mat");
save(modelFile, "net", "trainingInfo", "metrics", "classMetrics", "comparisonMetrics", ...
    "featureConfig", "featureNames", "featureMu", "featureSigma", "classNames", ...
    "layers", "options", "scoresTest", "YPredTest", "confusionCounts", ...
    "confusionTrain", "confusionVal");
writetable(comparisonMetrics, fullfile(modelsDir, "lstm_feature_comparison.csv"));
writetable(classMetrics, fullfile(modelsDir, "lstm_feature_class_metrics.csv"));
disp(comparisonMetrics);
disp(classMetrics);
fprintf("Training completed in %.1f seconds. Saved %s\n", trainingSeconds, modelFile);

function result = classificationMetrics(counts, classNames)
    truePositive = diag(counts);
    falseNegative = sum(counts, 2)-truePositive;
    falsePositive = sum(counts, 1)'-truePositive;
    precision = truePositive ./ max(truePositive+falsePositive, 1);
    recall = truePositive ./ max(truePositive+falseNegative, 1);
    f1Score = 2*precision.*recall ./ max(precision+recall, eps);
    result = table(classNames(:), truePositive, falsePositive, falseNegative, ...
        precision, recall, f1Score, VariableNames=["Class", "TruePositive", ...
        "FalsePositive", "FalseNegative", "Precision", "Recall", "F1Score"]);
end
