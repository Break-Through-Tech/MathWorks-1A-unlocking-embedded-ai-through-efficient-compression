%% Settings
clear
folder    = "compressed";                     % "uncompressed" or "compressed"
modelName = "daniel_cnn_optimized_pruned80_quantized";

%% Load test data
load('../../data/standardized.mat');
XTest  = num2cell(testStandardized', 1)';
YTest  = categorical(testLabels(:));
classNames = categories(categorical(trainLabels(:)));

%% Load model (works no matter what variable name it was saved under)
s = load("../../models/" + folder + "/" + modelName + ".mat");
f = fieldnames(s);
net = s.(f{1});

%% Results
YPred = scores2label(minibatchpredict(net, XTest), classNames);
testAccuracy = mean(YPred == YTest)
metrics = estimateNetworkMetrics(net);
learnables = sum(metrics.NumberOfLearnables)
modelSizeKB = sum(metrics.("ParameterMemory (MB)")) * 1024

%% Confusion matrix
figure;
confusionchart(YTest, YPred);
cm = confusionchart(YTest, YPred);
cm.Title = strrep(modelName, "_", " ") + " (test set)";
saveas(gcf, "../../docs/figures/" + modelName + "_confusion.png");