%% Load original model and data
S = load("trained_cnn_gru_model.mat");
netBaseline = S.net;

T = load(fullfile("data", "train.mat"));
V = load(fullfile("data", "val.mat"));

YVal = categorical(V.valLabels(:));
classNames = netBaseline.Layers(end).Classes;

% Convert to the network format required by projection.
netDL = dag2dlnetwork(netBaseline);
netDL = resetState(netDL);

%% Verify conversion preserves validation predictions
XVal = num2cell(single(V.valData), 2);
predOriginal = classify(netBaseline, XVal, ...
    "MiniBatchSize", 1, "ExecutionEnvironment", "cpu");

predConverted = predictLabels(netDL, V.valData, classNames);
assert(all(predOriginal == predConverted), ...
    "Conversion changed predictions. Stop and investigate.");

fprintf("Baseline validation accuracy: %.2f%%\n", ...
    100 * mean(predConverted == YVal));

%% Compress using TRAINING inputs only
% Format: channels x time steps x observations.
XTrain = dlarray(reshape(single(T.trainData.'), ...
    1, size(T.trainData, 2), size(T.trainData, 1)), "CTB");

[netProjected, projectionInfo] = compressNetworkUsingProjection( ...
    netDL, XTrain, ...
    ExplainedVarianceGoal=0.99, ...
    MiniBatchSize=16, ...
    UnpackProjectedLayers=true);

%% Compare validation accuracy and parameter storage
predProjected = predictLabels(netProjected, V.valData, classNames);

[nBefore, bytesBefore] = parameterStorage(netDL);
[nAfter, bytesAfter] = parameterStorage(netProjected);

fprintf("\nProjected validation accuracy: %.2f%%\n", ...
    100 * mean(predProjected == YVal));
fprintf("Parameters: %d -> %d\n", nBefore, nAfter);
fprintf("Parameter storage: %.2f -> %.2f KiB\n", ...
    bytesBefore / 1024, bytesAfter / 1024);
fprintf("Parameter-storage reduction: %.2f%%\n", ...
    100 * (1 - bytesAfter / bytesBefore));
fprintf("Share of 2 MiB Flash: %.2f%% -> %.2f%%\n", ...
    100 * bytesBefore / (2 * 1024^2), ...
    100 * bytesAfter / (2 * 1024^2));

disp(projectionInfo);

% Save a new file; leave the original model unchanged.
outputFile = "cnn_gru_projected_" + ...
    string(datetime("now", "Format", "yyyyMMdd_HHmmss")) + ".mat";
save(outputFile, "netProjected", "projectionInfo", "classNames");
fprintf("Saved: %s\n", outputFile);

%% Helper functions
function labels = predictLabels(network, data, classNames)
indices = zeros(size(data, 1), 1);
for i = 1:size(data, 1)
    x = dlarray(reshape(single(data(i, :)), ...
        1, size(data, 2), 1), "CTB");
    scores = predict(resetState(network), x);
    [~, indices(i)] = max(extractdata(scores(:)));
end
labels = classNames(indices);
labels = labels(:);
end

function [count, bytes] = parameterStorage(network)
count = 0;
bytes = 0;
for i = 1:height(network.Learnables)
    value = gather(extractdata(network.Learnables.Value{i}));
    details = whos("value");
    count = count + numel(value);
    bytes = bytes + details.bytes;
end
end