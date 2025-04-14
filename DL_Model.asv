clc; 
clear; 
close all;

% Load Data for Binary Classification

data = load('A01T.mat');
fs = 250;
window_length = 7*fs;
t1 = 0.5 * fs;  % Start time 
t2 = 3 * fs;    % Stop time
num_run = 6;

% Initialize
eeg_data = {};
labels = [];

% Extract trials from runs 4-9 (only for labels 1 and 2) (6 runs)
for i = 4:4+num_run-1
    trial_data = data.data{i};
    trials = trial_data.trial;
    y = trial_data.y;
    
    for t = 1:length(trials)
        % Only keep labels 1 (left) and 2 (right)
        if y(t) == 1 || y(t) == 2
            segment = trial_data.X(trials(t):trials(t)+window_length, 1:22);
            eeg_data = [eeg_data; {segment(t1:t2, :)}];
            labels = [labels; y(t)];
        end
    end
end

% Filter Data (4-40Hz Bandpass)
low_cutoff = 4;
high_cutoff = 40;
order = 4;

[b, a] = butter(order, [low_cutoff, high_cutoff] / (fs/2), 'bandpass');
selected_labels = [1,2];
selected_indices = ismember(labels, selected_labels);

% Apply the filter to only selected trials
filtered_data = [];
for i = 1:length(eeg_data)
    if selected_indices(i)
        % Apply zero-phase filtering to preserve signal shape
        f = filtfilt(b, a, eeg_data{i});
        filtered_data = [filtered_data; {f}];
    end
end

% Convert the data to 4D array of shape [samples x channels x 1 X trials]
X = zeros(t2-t1+1, 22, 1, length(filtered_data));
for i = 1:length(filtered_data)
    X(:,:,:,i) = filtered_data{i};
end

disp(size(X));

% Convert them into 1=left and 2=right for binary classification
Y = categorical(labels);

% Verify if there are only 2 classes
if numel(unique(labels)) ~= 2
    error('More than 2 classes!!!')
end

% Train-Test split (80-20)
rng(42);
cv = cvpartition(Y, 'HoldOut', 0.2);
trainX = X(:,:,:,cv.training);
trainY = Y(cv.training);
testX = X(:,:,:,cv.test);
testY = Y(cv.test);

% Attempting EEGNet Architecture
layers = [
    imageInputLayer([t2-t1+1 22 1], 'Name', 'input', 'Normalization', 'zscore')
    
    % Temporal Convolution
    convolution2dLayer([1 64], 8, 'Padding', 'same', 'Name', 'temp_conv')
    batchNormalizationLayer('Name', 'bh1')
    eluLayer('Name', 'elu1')
    dropoutLayer(0.25, 'Name', 'drop1')
    
    % Spatial convolution
    convolution2dLayer([22 1], 16, 'Padding', 'same', 'Name', 'spat_conv')
    batchNormalizationLayer('Name', 'bn2')
    eluLayer('Name', 'elu2')
    dropoutLayer(0.25, 'Name', 'drop2')
    maxPooling2dLayer([4 1], 'Stride', [4 1], 'Name', 'pool1')
    
    % Separable convolution
    convolution2dLayer([1 16], 16, 'Padding', 'same', 'Name', 'sep_conv')
    batchNormalizationLayer('Name', 'bn3')
    eluLayer('Name', 'elu3')
    dropoutLayer(0.25, 'Name', 'drop3')
    
    % Classifier
    fullyConnectedLayer(32, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    
    dropoutLayer(0.5, 'Name', 'drop4')
    fullyConnectedLayer(2, 'Name', 'fc2')  % Only 2 output units for binary
    
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Training Options
options = trainingOptions('adam', 'MaxEpochs', 60, 'MiniBatchSize', 32, 'InitialLearnRate', 0.0005, 'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', 0.5, 'LearnRateDropPeriod', 20, 'L2Regularization', 0.001, 'ValidationData', {testX, testY}, 'ValidationFrequency', 30, ...
            'Shuffle', 'every-epoch', 'Plots', 'training-progress', 'Verbose', true);

% Train Network
net = trainNetwork(trainX, trainY, layers, options);

% Evaluation
[predicted, scores] = classify(net, testX);

% Metrics
accuracy = mean(predicted == testY);
[precision, recall, F1] = calculateMetrics(predicted, testY);

fprintf('Results:\n');
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f%%\n', precision* 100);
fprintf('Recall: %.2f%%\n', recall * 100);
fprintf('F1 Score: %.2f%%\n', F1*100);


% Helper functions
function [precision, recall, F1] = calculateMetrics(predicted, trueLabels)
    % Convert to numeric
    pred = double(predicted);
    true = double(trueLabels);
    
    % Getting TN, FP, FN, TP
    TP = sum((pred == 2) & (true == 2));
    FP = sum((pred == 2) & (true == 1));
    FN = sum((pred == 1) & (true == 2));
    TN = sum((pred == 1) & (true == 1));
    
    % Calculating precision, recall, F1
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    F1 = 2 * (precision * recall) / (precision + recall);
end