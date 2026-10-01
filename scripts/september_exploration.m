% Week 1 - Data Exploration
clear
load('../data/train.mat')

% Shape and class balance
size(trainData)
unique(trainLabels)
groupcounts(trainLabels)

% Data quality: missing/infinite values, overall range
any(isnan(trainData(:)))
any(isinf(trainData(:)))
min(trainData(:))
max(trainData(:))

% RMS (size of vibration) and kurtosis (how much spikes stand out)
% , per recording
allRMS = rms(trainData, 2);
allKurt = kurtosis(trainData, [], 2);

% Average them per class
classes = unique(trainLabels);
for k = 1:3
    rows = strcmp(trainLabels, classes{k});
    fprintf('%s: RMS=%.3f  Kurtosis=%.2f\n', classes{k}, ...
        mean(allRMS(rows)), mean(allKurt(rows)));
end

% Plot 1: one recording per class, same scale
figure
for k = 1:3
    idx = find(strcmp(trainLabels, classes{k}), 1);
    subplot(3,1,k)
    plot(trainData(idx,:))
    ylim([-25 25])
    title(classes{k})
end
set(gcf, 'Position', [100 100 600 800])
% Uncomment to save image, you're free to change the save path as well
% exportgraphics(gcf, '../docs/figures/all_three_classes.png', 'BackgroundColor', 'current')

% Plot 2: Normal vs OuterRaceFault on a tighter scale
figure
subplot(2,1,1)
idx = find(strcmp(trainLabels, 'Normal'), 1);
plot(trainData(idx,:))
ylim([-6 6])
title('Normal')

subplot(2,1,2)
idx = find(strcmp(trainLabels, 'OuterRaceFault'), 1);
plot(trainData(idx,:))
ylim([-6 6])
title('OuterRaceFault')
set(gcf, 'Position', [100 100 600 700])
% Uncomment to save image, you're free to change the save path as well
% exportgraphics(gcf, '../docs/figures/normal_vs_outer.png', 'BackgroundColor', 'current')