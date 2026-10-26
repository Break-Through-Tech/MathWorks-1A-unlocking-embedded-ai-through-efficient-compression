%% Settings
clear
seed     = 0;
loadName = "daniel_cnn_optimized";
goals    = [0.3 0.5 0.6 0.7 0.8 0.9 0.95];   % compression levels to try
fineTuneEpochs = 10;

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

%% Load trained CNN
net = load("../../models/uncompressed/" + loadName + ".mat").net;
metrics = estimateNetworkMetrics(net);
origLearnables = sum(metrics.NumberOfLearnables);

pruneOptions = trainingOptions("adam", ...
    MaxEpochs=fineTuneEpochs, ...
    MiniBatchSize=32, ...
    Shuffle="every-epoch", ...
    ValidationData={XVal, YVal}, ...
    Metrics="accuracy", ...
    Verbose=false);

%% Prune at each level
results = table();

for g = goals
    rng(seed);
    netPruned = compressNetworkUsingTaylorPruning(net, XTrain, YTrain, ...
        "crossentropy", pruneOptions, LearnablesReductionGoal=g);

    YPred = scores2label(minibatchpredict(netPruned, XTest), classNames);
    m = estimateNetworkMetrics(netPruned);
    learnables = sum(m.NumberOfLearnables);

    results = [results; table(g*100, ...
        (1 - learnables/origLearnables)*100, ...
        mean(YPred == YTest)*100, ...
        learnables, ...
        sum(m.("ParameterMemory (MB)"))*1024, ...
        VariableNames=["GoalPct", "ActualCompressionPct", "TestAccPct", ...
        "Learnables", "ModelSizeKB"])];

    save("../../models/compressed/" + loadName + "_pruned" + round(g*100) + ".mat", 'netPruned');
end

disp(results)

%% Compression vs accuracy curve
figure;
plot(results.ActualCompressionPct, results.TestAccPct, '-o', LineWidth=1.5);
xlabel("Compression (% of learnables removed)");
ylabel("Test accuracy (%)");
title(loadName + ": compression vs. accuracy");
grid on;
saveas(gcf, "../../docs/figures/" + loadName + "_compression_curve.png");