% Load without overwriting the original saved model.
S = load("trained_cnn_gru_model.mat");
netBaseline = S.net;

T = load(fullfile("data", "train.mat"));
V = load(fullfile("data", "val.mat"));

% Compute normalization statistics using TRAINING data only.
mu = mean(T.trainData(:));
sigma = std(T.trainData(:));
assert(sigma > 0, "Training standard deviation must be positive.");

% Two possible input preparations.
XValRaw = num2cell(single(V.valData), 2);
XValStandardized = num2cell( ...
    single((V.valData - mu) / sigma), 2);

YVal = categorical(V.valLabels(:));

% Evaluate the unchanged model.
predRaw = classify(netBaseline, XValRaw, ...
    "MiniBatchSize", 1, "ExecutionEnvironment", "cpu");

predStandardized = classify(netBaseline, XValStandardized, ...
    "MiniBatchSize", 1, "ExecutionEnvironment", "cpu");

fprintf("Training mean: %.8f\n", mu);
fprintf("Training standard deviation: %.8f\n\n", sigma);

fprintf("Raw inputs: %d/%d correct (%.2f%%)\n", ...
    sum(predRaw == YVal), numel(YVal), ...
    100 * mean(predRaw == YVal));

fprintf("Globally standardized inputs: %d/%d correct (%.2f%%)\n", ...
    sum(predStandardized == YVal), numel(YVal), ...
    100 * mean(predStandardized == YVal));