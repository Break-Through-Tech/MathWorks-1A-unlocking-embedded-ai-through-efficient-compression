%% Settings
clear
folder    = "compressed";                     % "uncompressed" or "compressed"
modelName = "daniel_hybrid_lstm_gru_proj70_quantized";
bytesPerWeight = 1;                           % 1 if quantized (int8), 4 if not

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
learnables = sum(cellfun(@numel, net.Learnables.Value))   
modelSizeKB = learnables * bytesPerWeight / 1024

%% Confusion matrix
figure;
cm = confusionchart(YTest, YPred);
cm.Title = strrep(modelName, "_", " ") + " (test set)";
saveas(gcf, "../../docs/figures/" + modelName + "_confusion.png");