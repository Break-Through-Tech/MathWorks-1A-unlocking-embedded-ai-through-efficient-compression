%% Settings
clear
loadName = "daniel_cnn_optimized_pruned80";
saveName = loadName + "_quantized";

%% Load data
load('../../data/standardized.mat');
XTest  = num2cell(testStandardized', 1)';
YTest  = categorical(testLabels(:));
classNames = categories(categorical(trainLabels(:)));

%% Load pruned CNN
net = load("../../models/compressed/" + loadName + ".mat").netPruned;

%% Calibration data (channels x time x recordings)
X3 = reshape(trainStandardized', 1, 5000, []);
calData = dlarray(X3, "CTB");

%% Quantize
quantObj = dlquantizer(net, ExecutionEnvironment="MATLAB");
calibrate(quantObj, calData);
qNet = quantize(quantObj);

%% Results
YPred = scores2label(minibatchpredict(qNet, XTest), classNames);
testAccuracy = mean(YPred == YTest)
metrics = estimateNetworkMetrics(qNet);
learnables = sum(metrics.NumberOfLearnables)
modelSizeKB = sum(metrics.("ParameterMemory (MB)")) * 1024

%% Save
save("../../models/compressed/" + saveName + ".mat", 'qNet');