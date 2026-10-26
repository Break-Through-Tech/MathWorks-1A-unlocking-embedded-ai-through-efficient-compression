%% Settings
clear
seed     = 0;
loadName = "daniel_hybrid_lstm_gru";
goals    = [0.3 0.5 0.6 0.7 0.8 0.9];   % compression levels to try
fineTuneEpochs = 5;

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

% projection data: channels x time x recordings
projData = dlarray(reshape(trainStandardized', 1, 5000, []), "CTB");

%% Load trained hybrid
net = load("../../models/uncompressed/" + loadName + ".mat").net;
origLearnables = sum(cellfun(@numel, net.Learnables.Value));

ftOptions = trainingOptions("adam", ...
    MaxEpochs=fineTuneEpochs, ...
    MiniBatchSize=32, ...
    Shuffle="every-epoch", ...
    ValidationData={XVal, YVal}, ...
    Metrics="accuracy", ...
    GradientThreshold=1, ...
    Verbose=false);

if ~isfolder("../../models/compressed"), mkdir("../../models/compressed"); end

%% Project at each level
results = table();

for g = goals
    rng(seed);
    netProj = compressNetworkUsingProjection(net, projData, ...
        LearnablesReductionGoal=g, VerbosityLevel="off");

    % accuracy right after projection
    YPred = scores2label(minibatchpredict(netProj, XTest), classNames);
    accBefore = mean(YPred == YTest) * 100;

    % fine-tune to recover accuracy
    netProj = trainnet(XTrain, YTrain, netProj, "crossentropy", ftOptions);
    YPred = scores2label(minibatchpredict(netProj, XTest), classNames);
    accAfter = mean(YPred == YTest) * 100;

    learnables = sum(cellfun(@numel, netProj.Learnables.Value));

    results = [results; table(g*100, ...
        (1 - learnables/origLearnables)*100, ...
        accBefore, accAfter, learnables, learnables*4/1024, ...
        VariableNames=["GoalPct", "ActualCompressionPct", ...
        "AccBeforeFT", "AccAfterFT", "Learnables", "ModelSizeKB"])];

    save("../../models/compressed/" + loadName + "_proj" + round(g*100) + ".mat", 'netProj');
end

disp(results)

%% Compression vs accuracy curve
figure;
plot(results.ActualCompressionPct, results.AccBeforeFT, '-o', LineWidth=1.5); hold on;
plot(results.ActualCompressionPct, results.AccAfterFT, '-s', LineWidth=1.5);
legend("Before fine-tuning", "After fine-tuning", Location="southwest");
xlabel("Compression (% of learnables removed)");
ylabel("Test accuracy (%)");
title(strrep(loadName, "_", " ") + ": compression vs. accuracy");
ylim([0 100.5]);
grid on;
saveas(gcf, "../../docs/figures/" + loadName + "_compression_curve.png");