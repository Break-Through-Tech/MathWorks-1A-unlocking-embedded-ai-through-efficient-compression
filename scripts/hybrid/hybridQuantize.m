%% Settings
clear
loadName = "daniel_hybrid_lstm_gru_proj70";
saveName = loadName + "_quantized";

%% Load data
load('../../data/standardized.mat');
XTest  = num2cell(testStandardized', 1)';
YTest  = categorical(testLabels(:));
classNames = categories(categorical(trainLabels(:)));

%% Load projected hybrid
net = load("../../models/compressed/" + loadName + ".mat").netProj;
learnables = sum(cellfun(@numel, net.Learnables.Value))

%% Calibration data (channels x time x recordings)
calData = dlarray(reshape(trainStandardized', 1, 5000, []), "CTB");

%% Quantize
quantObj = dlquantizer(net, ExecutionEnvironment="MATLAB");
prepareNetwork(quantObj);               % fuses batchnorm into conv
calibrate(quantObj, calData);
qNet = quantize(quantObj);
%% Results
YPred = scores2label(minibatchpredict(qNet, XTest), classNames);
testAccuracy = mean(YPred == YTest)
modelSizeKB = learnables * 1 / 1024     % int8 = 1 byte per weight (estimate)

%% Save
save("../../models/compressed/" + saveName + ".mat", 'qNet');