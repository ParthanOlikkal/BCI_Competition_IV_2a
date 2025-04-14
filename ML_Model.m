clc;
clear;
close all;

% Load Data
data = load('A03T.mat');
fs = 250;
num_run = 6;
window_length = 7*fs;
t1 = 0.5 * fs;  % start time
t2 = 3 * fs;    % stop time

eeg_data = {};
labels = [];

% Extract trials from runs 4-9 (6 runs)
for i = 4:4+num_run-1
    file = data.data{i};
    X = file.X;
    y = file.y;
    trials = file.trial;
    
    for t = 1:length(trials)
        % Extract relevant time window and channels (1:22)
        temp = X(trials(t):trials(t)+window_length, 1:22);
        eeg_data = [eeg_data; {temp(t1:t2, :)}];
        labels = [labels; y(t)];
    end
end

% Looking at left and right class for binary classification
left_data = eeg_data(labels == 1);
right_data = eeg_data(labels == 2);

% Create labels (1 for left, 2 for right)
left_labels = ones(length(left_data), 1);
right_labels = 2 * ones(length(right_data), 1);

% Filter Bank
subbands = [4 8; 8 12; 12 16; 16 20; 20 24; 24 28; 28 32; 32 36; 36 40];
num_bands = size(subbands, 1);
num_channels = size(eeg_data{1}, 2);

% FBCSP Feature Extraction
[m_features, m_labels] = extractFBCSPFeatures(left_data, right_data, left_labels, right_labels, subbands, fs);


% Mutual Information Feature selection
[m_features, mu, sigma] = zscore(m_features);

% Selecting tion 10 features using (Minimum Redundancy Maximum Relevance) MRMR
[ranked_idx, ~] = fscmrmr(m_features, m_labels);
selected_features = ranked_idx(1:10);

m_features = m_features(:, selected_features);

% Train test split (80: 20)
[trainX, testX, trainY, testY] = train_test_split(m_features, m_labels, 0.8);

% SVM
svm_model = fitcsvm(trainX, trainY, 'KernelFunction', 'rbf', 'Standardize', true, 'BoxConstraint', 1, 'KernelScale', 'auto');


% Evaluate
trainPred = predict(svm_model, trainX);
testPred = predict(svm_model, testX);

trainAcc = sum(trainPred == trainY) / length(trainY) * 100;
testAcc = sum(testPred == testY) / length(testY) * 100;

fprintf("Training Accuracy: %.2f%%\n", trainAcc);
fprintf("Testing Accuracy: %.2f%%\n", testAcc);

% Confusion Matrix
confusionchart(testY, testPred);
title('Confusion Matrix');



% Helper function
function [trainX, testX, trainY, testY] = train_test_split(X, y, split_ratio)
    % Shuffle data
    rng(42);
    idx = randperm(size(X,1));
    X = X(idx,:);
    y = y(idx);
    
    split_point = round(split_ratio * size(X, 1));
    trainX = X(1:split_point, :);
    testX = X(split_point+1:end, :);
    trainY = y(1:split_point);
    testY = y(split_point+1:end);
end

function [features, labels] = extractFBCSPFeatures(left_data, right_data, left_label, right_label, subbands, fs)
    num_bands = size(subbands, 1);
    num_trials_left = length(left_data);
    num_trials_right = length(right_data);
    num_trials = num_trials_left + num_trials_right;
    num_channels = size(left_data{1}, 2);
    
    % Assuming 4 CSP features per band
    features = zeros(num_trials, num_bands * 4);
    labels = [left_label; right_label];
    
    % Process each frequency band
    for band = 1:num_bands
        low_cut = subbands(band, 1);
        high_cut = subbands(band, 2);
        
        % Filter both classes
        left_filtered = filterTrials(left_data, fs, low_cut, high_cut);
        right_filtered = filterTrials(right_data, fs, low_cut, high_cut);
        
        % Compute CSP for the band
        [W, lambda] = csp(left_filtered, right_filtered);
        
        % Extract CSP features (as mentioned first and last 2 components)
        csp_indices  = [1:2, (size(W, 1)-1):size(W, 1)];
        
        % Extract Features
        for t = 1:num_trials_left
            trial = left_filtered{t};
            proj = trial * W(:, csp_indices);   % Projection
            features(t, (band-1)*4+(1:4)) = log(var(proj));
        end
        
        for t = 1:num_trials_right
            trial = right_filtered{t};
            proj = trial * W(:, csp_indices);
            features(num_trials_left+t, (band-1)*4+(1:4)) = log(var(proj));
        end
    end
end

function [W, lambda] = csp(left_data, right_data)
    % Compute Covariance
    R1 = zeros(size(left_data{1}, 2));
    for t = 1:length(left_data)
        trial = left_data{t};
        R1 = R1 + cov(trial);
    end
    R1 = R1 / length(left_data);
    
    R2 = zeros(size(right_data{1}, 2));
    for t = 1:length(right_data)
        trial = right_data{t};
        R2 = R2 + cov(trial);
    end
    R2 = R2 / length(right_data);
    
    % Eigenvalue problem
    [W, D] = eig(R1, R1+R2);
    lambda = diag(D);
    
    % Sort eigen vectors in descending order
    [lambda, idx] = sort(lambda, 'descend');
    W = W(:, idx);
end

function filtered_trials = filterTrials(trials, fs, low_cut, high_cut)
    filtered_trials = cell(size(trials));
    [b, a] = butter(4, [low_cut, high_cut] / (fs/2), 'bandpass');
    
    for t = 1:length(trials)
        trial = trials{t};
        filtered_trial = zeros(size(trial));
        for ch = 1:size(trial, 2)
            filtered_trial(:,ch) = filtfilt(b,a, trial(:, ch));
        end
        filtered_trials{t} = filtered_trial;
    end
end