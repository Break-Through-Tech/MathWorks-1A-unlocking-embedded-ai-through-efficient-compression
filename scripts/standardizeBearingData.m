clear
clc

%% Load the datasets

load("train.mat")
load("val.mat")
load("test.mat")

%% Calculate statistics using training data only

mu = mean(trainData(:));
sigma = std(trainData(:));

%% Standardize all datasets

trainDataStandardized = (trainData - mu) / sigma;
valDataStandardized = (valData - mu) / sigma;
testDataStandardized = (testData - mu) / sigma;

%% Display standardization results

disp("Training data:")
disp(["Mean: " + mean(trainDataStandardized(:)), ...
    " Standard deviation: " + std(trainDataStandardized(:))])

disp("Validation data:")
disp(["Mean: " + mean(valDataStandardized(:)), ...
    " Standard deviation: " + std(valDataStandardized(:))])

disp("Test data:")
disp(["Mean: " + mean(testDataStandardized(:)), ...
    " Standard deviation: " + std(testDataStandardized(:))])

%% Compare data ranges before and after standardization

disp("Original training-data range:")
disp([min(trainData(:)), max(trainData(:))])

disp("Standardized training-data range:")
disp([min(trainDataStandardized(:)), ...
    max(trainDataStandardized(:))])

disp("Original validation-data range:")
disp([min(valData(:)), max(valData(:))])

disp("Standardized validation-data range:")
disp([min(valDataStandardized(:)), ...
    max(valDataStandardized(:))])

disp("Original test-data range:")
disp([min(testData(:)), max(testData(:))])

disp("Standardized test-data range:")
disp([min(testDataStandardized(:)), ...
    max(testDataStandardized(:))])

%% Save the standardized datasets

save("standardizedBearingData.mat", ...
    "trainDataStandardized", ...
    "valDataStandardized", ...
    "testDataStandardized", ...
    "trainLabels", ...
    "valLabels", ...
    "testLabels", ...
    "mu", ...
    "sigma");

disp("Standardization completed successfully.")

%% Visualize a Normal signal before and after standardization

% Find the first Normal signal
normalIndex = find(strcmp(string(trainLabels), "Normal"), 1);

originalSignal = trainData(normalIndex,:);
standardizedSignal = trainDataStandardized(normalIndex,:);

% Use the same vertical scale for both plots
commonLimits = [
    min([originalSignal, standardizedSignal]), ...
    max([originalSignal, standardizedSignal])
    ];

figure

subplot(2,1,1)
plot(originalSignal)
title("Normal Signal Before Standardization")
xlabel("Sample Number")
ylabel("Original Value")
ylim(commonLimits)
grid on

subplot(2,1,2)
plot(standardizedSignal)
title("Same Normal Signal After Standardization")
xlabel("Sample Number")
ylabel("Standardized Value")
ylim(commonLimits)
grid on

sgtitle("Signal Before and After Standardization")