function features = extract_bearing_features(data, config)
%EXTRACT_BEARING_FEATURES Convert N-by-5000 signals to T-by-4-by-N features.
% Each frame is demeaned. Channels are RMS, Pearson kurtosis (Gaussian = 3),
% crest factor, and Hann-windowed power fraction above config.bandStartHz.
% No statistics are fitted here; feature scaling belongs to the training set.

arguments
    data (:,:) {mustBeNumeric, mustBeReal, mustBeFinite}
    config (1,1) struct
end

validateattributes(config.signalLength, {'numeric'}, {'scalar','integer','positive'});
validateattributes(config.windowLength, {'numeric'}, {'scalar','integer','>=',4});
validateattributes(config.hopLength, {'numeric'}, {'scalar','integer','positive'});
validateattributes(config.sampleRate, {'numeric'}, {'scalar','finite','positive'});
validateattributes(config.bandStartHz, {'numeric'}, ...
    {'scalar','finite','positive','<',config.sampleRate/2});
assert(size(data, 2) == config.signalLength, ...
    'BearingFeatures:SignalLength', 'Each row must contain %d samples.', config.signalLength);
assert(~isempty(data), 'BearingFeatures:EmptyData', 'Supply at least one signal.');
assert(config.windowLength <= config.signalLength && ...
    config.hopLength <= config.windowLength, ...
    'BearingFeatures:WindowLength', 'Frames must fit the signal and leave no gaps.');
assert(mod(config.signalLength-config.windowLength, config.hopLength) == 0, ...
    'BearingFeatures:Coverage', 'The final frame must end at the final sample.');

windowLength = config.windowLength;
frameStarts = 1:config.hopLength:(size(data, 2)-windowLength+1);
features = zeros(numel(frameStarts), 4, size(data, 1), 'single');
taper = 0.5 - 0.5*cos(2*pi*(0:windowLength-1)/(windowLength-1));
numBins = floor(windowLength/2) + 1;
frequencies = (0:numBins-1)*config.sampleRate/windowLength;
highBand = frequencies >= config.bandStartHz;

for t = 1:numel(frameStarts)
    frame = double(data(:, frameStarts(t)+(0:windowLength-1)));
    frame = frame - mean(frame, 2);
    frameRMS = sqrt(mean(frame.^2, 2));
    denominator = frameRMS;
    denominator(denominator == 0) = 1;
    normalized = frame ./ denominator;
    frameKurtosis = mean(normalized.^4, 2);
    crestFactor = max(abs(normalized), [], 2);

    spectrum = fft(frame .* taper, [], 2);
    powerSpectrum = abs(spectrum(:, 1:numBins)).^2;
    % Account for negative-frequency energy; DC and Nyquist are unpaired.
    if mod(windowLength, 2) == 0
        powerSpectrum(:, 2:end-1) = 2*powerSpectrum(:, 2:end-1);
    else
        powerSpectrum(:, 2:end) = 2*powerSpectrum(:, 2:end);
    end
    totalPower = sum(powerSpectrum, 2);
    totalPower(totalPower == 0) = 1;
    highBandFraction = sum(powerSpectrum(:, highBand), 2) ./ totalPower;

    frameFeatures = [frameRMS, frameKurtosis, crestFactor, highBandFraction];
    features(t, :, :) = permute(single(frameFeatures), [3 2 1]);
end
end
