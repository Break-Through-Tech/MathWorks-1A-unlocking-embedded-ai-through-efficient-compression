function quantizationResults = calibrate_feature_lstm_int8
%CALIBRATE_FEATURE_LSTM_INT8 Calibrate and, when possible, quantize feature LSTMs.
% Calibrated dlquantizer objects are not replacements for trained networks.
% Creating a simulatable target-agnostic INT8 network requires Fixed-Point
% Designer. Code-generation support also depends on the execution environment,
% operating system, target library, and layers supported by the MATLAB release.

repoRoot = fileparts(fileparts(fileparts(mfilename("fullpath"))));
modelsDir = fullfile(repoRoot, "models");
compressedModelsDir = fullfile(modelsDir, "compressed");
if ~isfolder(compressedModelsDir)
    mkdir(compressedModelsDir);
end
baselineFile = fullfile(modelsDir, "uncompressed", ...
    "alex_feature_lstm_64.mat");
if ~isfile(baselineFile)
    error("Run train_feature_lstm_fault_classifier.m before calibration.");
end

hiddenUnitCandidates = [48 32];
candidateModelFiles = strings(numel(hiddenUnitCandidates), 1);
quantizerFiles = strings(numel(hiddenUnitCandidates), 1);
calibrationSucceeded = false(numel(hiddenUnitCandidates), 1);
calibrationMessage = strings(numel(hiddenUnitCandidates), 1);

baseline = load(baselineFile, "featureConfig", "featureNames", "featureMu", ...
    "featureSigma", "classNames");
trainSet = load(fullfile(repoRoot, "data", "train.mat"));
featuresTrain = extract_bearing_features(trainSet.trainData, baseline.featureConfig);
XTrain = (featuresTrain - baseline.featureMu) ./ baseline.featureSigma;
calibrationCount = min(128, size(XTrain, 3));
calibrationIndices = unique(round(linspace(1, size(XTrain, 3), ...
    calibrationCount)));
dlXCalibration = dlarray(XTrain(:, :, calibrationIndices), "TCB");

testSet = load(fullfile(repoRoot, "data", "test.mat"));
featuresTest = extract_bearing_features(testSet.testData, baseline.featureConfig);
XTest = (featuresTest - baseline.featureMu) ./ baseline.featureSigma;
YTest = categorical(string(testSet.testLabels), baseline.classNames);
executionEnvironments = strings(numel(hiddenUnitCandidates), 1);
simulationSucceeded = false(numel(hiddenUnitCandidates), 1);
quantizedTestAccuracy = nan(numel(hiddenUnitCandidates), 1);
quantizedLayerCount = zeros(numel(hiddenUnitCandidates), 1);
quantizedLearnablesBytes = nan(numel(hiddenUnitCandidates), 1);
simulationMessage = strings(numel(hiddenUnitCandidates), 1);

if ~(exist("dlquantizer", "class") == 8 || exist("dlquantizer", "file") == 2)
    calibrationMessage(:) = "dlquantizer is not installed";
else
    executionEnvironment = "CPU";
    if ismac
        % CPU and GPU code-generation environments are not available on macOS.
        executionEnvironment = "MATLAB";
    end

    for i = 1:numel(hiddenUnitCandidates)
        numHiddenUnits = hiddenUnitCandidates(i);
        candidateModelFiles(i) = fullfile(modelsDir, ...
            sprintf("lstm_feature_%d_fault_classifier.mat", numHiddenUnits));
        if ~isfile(candidateModelFiles(i))
            calibrationMessage(i) = "candidate model file not found";
            continue;
        end

        try
            candidate = load(candidateModelFiles(i), "net");
            dq = dlquantizer(candidate.net, ...
                ExecutionEnvironment=executionEnvironment);
            executionEnvironments(i) = executionEnvironment;
            calibrationResults = calibrate(dq, dlXCalibration, ...
                MiniBatchSize=32, UseGPU="off");
            quantizerFiles(i) = fullfile(modelsDir, ...
                sprintf("lstm_feature_%d_quantizer.mat", numHiddenUnits));
            calibrationResultsFile = fullfile(modelsDir, ...
                sprintf("lstm_feature_%d_calibration_results.mat", numHiddenUnits));
            save(quantizerFiles(i), "dq");
            save(calibrationResultsFile, "calibrationResults", ...
                "calibrationIndices", "executionEnvironment");
            calibrationSucceeded(i) = true;
            calibrationMessage(i) = "calibrated using " + executionEnvironment;

            if executionEnvironment == "MATLAB" && ...
                    exist("fi", "file") == 2 && ...
                    license("test", "Fixed_Point_Toolbox")
                quantizedNet = quantize(dq);
                details = quantizationDetails(quantizedNet);
                quantizedScores = minibatchpredict(quantizedNet, XTest, ...
                    MiniBatchSize=32, ExecutionEnvironment="cpu");
                quantizedLabels = scores2label(quantizedScores, ...
                    baseline.classNames);
                quantizedTestAccuracy(i) = 100*mean(quantizedLabels == YTest);
                quantizedLayerCount(i) = numel(details.QuantizedLayerNames);
                quantizedLearnablesBytes(i) = calculateQuantizedLearnablesBytes( ...
                    details.QuantizedLearnables);
                quantizedNetworkFile = fullfile(modelsDir, ...
                    sprintf("lstm_feature_%d_quantized_network.mat", ...
                    numHiddenUnits));
                if numHiddenUnits == 32
                    quantizedNetworkFile = fullfile(compressedModelsDir, ...
                        "alex_feature_lstm_32_quantized.mat");
                end
                qNet = quantizedNet;
                featureConfig = baseline.featureConfig;
                featureNames = baseline.featureNames;
                featureMu = baseline.featureMu;
                featureSigma = baseline.featureSigma;
                classNames = baseline.classNames;
                save(quantizedNetworkFile, "qNet", "details", ...
                    "featureConfig", "featureNames", "featureMu", ...
                    "featureSigma", "classNames");
                simulationSucceeded(i) = true;
                simulationMessage(i) = "simulatable INT8 network created";
            elseif executionEnvironment ~= "MATLAB"
                simulationMessage(i) = ...
                    "calibrated for target code generation; no MATLAB INT8 simulation";
            else
                simulationMessage(i) = ...
                    "Fixed-Point Designer is required for MATLAB INT8 simulation";
            end
        catch exception
            calibrationMessage(i) = string(exception.message);
            warning("FeatureLSTM:QuantizationCalibration", ...
                "Could not calibrate the %d-unit network: %s", ...
                numHiddenUnits, exception.message);
        end
    end
end

quantizationResults = table(hiddenUnitCandidates(:), executionEnvironments, ...
    calibrationSucceeded, calibrationMessage, simulationSucceeded, ...
    quantizedTestAccuracy, quantizedLayerCount, quantizedLearnablesBytes, ...
    quantizedLearnablesBytes/1024, simulationMessage, ...
    candidateModelFiles, quantizerFiles, ...
    VariableNames=["HiddenUnits", "ExecutionEnvironment", ...
    "CalibrationSucceeded", "CalibrationMessage", "SimulationSucceeded", ...
    "QuantizedTestAccuracy", "QuantizedLayerCount", ...
    "QuantizedLearnablesBytes", "QuantizedLearnablesKiB", ...
    "SimulationMessage", "ModelFile", "QuantizerFile"]);
writetable(quantizationResults, ...
    fullfile(modelsDir, "lstm_feature_quantization_results.csv"));
save(fullfile(modelsDir, "lstm_feature_quantization_results.mat"), ...
    "quantizationResults");
compressionResultsFile = fullfile(modelsDir, ...
    "lstm_feature_compression_results.mat");
if isfile(compressionResultsFile)
    savedResults = load(compressionResultsFile);
    savedResults.quantizationResults = quantizationResults;
    save(compressionResultsFile, "-struct", "savedResults");
end
disp(quantizationResults);
end

function totalBytes = calculateQuantizedLearnablesBytes(learnables)
totalBits = 0;
for k = 1:height(learnables)
    value = learnables.Value{k};
    dataType = numerictype(value);
    totalBits = totalBits + numel(value)*dataType.WordLength;
end
totalBytes = ceil(totalBits/8);
end
