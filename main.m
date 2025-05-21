% No Transfer Learning Yet

clc;
clear;
close(findall(0,'Type','figure'))

% Check GPU availability
useGPU = false;
if gpuDeviceCount > 0
    gpu = gpuDevice(1); % Use the first GPU
    fprintf('Using GPU: %s with %.2f GB memory\n', gpu.Name, gpu.AvailableMemory/1e9);
    reset(gpu); % Reset GPU memory
    useGPU = true;
else
    fprintf('No GPU available. Using CPU for computations.\n');
end

% System parameters
m = 0.5; % kg
k = 217; % N/m
kp = 63.5; % N/m^3
d = 0.25; % N⋅s/m

% Simulation parameters
sim_time = 60; % s
dt = 0.01; % s

% Initial conditions
x0 = [0, 0, 0]; % m
xdot0 = [0, 0, 0]; % m/s

% Input
max_force_list = [10, 50, 100, 500];

% Split ratios
trainRatio = 0.8;
valRatio = 0.1;
testRatio = 0.1;

% Storage for sequences by dataset type
XTrain = {};
YTrain = {};
XVal = {};
YVal = {};
XTest = {};
YTest = {};

% Sequence parameters
seq_length = 100;
step_size = 50;

for i = 1:length(max_force_list)
    max_force = max_force_list(i);
    u1 = @(t) 0;
    dist = {@(t) max_force, @(t) max_force*sin(t), @(t) max_force*(t<0.1), @(t) max_force*sin(2*pi*(0.1+(10-0.1)*t/20)*t), @(t) max_force*cos(2*pi*2500*t) .* randn(size(t)) .* (rand(size(t)) > 0.5), @(t) max_force * (sin(t) + sin(2*t)/sqrt(2) + sin(4*t)/2 + sin(8*t)/sqrt(8) + sin(16*t)/4) ./ 2, @(t) max_force * ( ...
        sin(997*t) + sin(1009*t) + sin(1013*t) + sin(1019*t) + ...
        sin(1021*t) + sin(1031*t) + sin(1033*t) + sin(1039*t) + ...
        sin(1049*t) + sin(1051*t) + sin(1061*t) + sin(1063*t) + ...
        sin(1069*t) + sin(1087*t) + sin(1091*t) + sin(1093*t) ...
        ) / 16.0};
    u3 = @(t) 0;
    
    for j = 1:length(dist)
        % For each simulation type (force level + disturbance function)
        dist2 = dist{1,j};
        [t, x, xdot, y] = simulate_msd(sim_time, dt, u1, dist2, u3, m, k, kp, d, x0, xdot0);
        dist_values = arrayfun(dist2, t)';
        my_in = [x xdot dist_values];
        my_out = [x xdot];
        [X, Y] = slice_sequences(my_in, my_out, seq_length, step_size);
        
        fprintf("Simulation %d-%d done: %d sequences\n", i, j, length(X));
        
        % Split sequences for this specific simulation
        numSequences = length(X);
        indices = randperm(numSequences); % Shuffle sequences for this simulation
        
        % Calculate split indices
        trainEnd = floor(trainRatio * numSequences);
        valEnd = trainEnd + floor(valRatio * numSequences);
        
        % Split into train/val/test for this simulation
        trainIndices = indices(1:trainEnd);
        valIndices = indices(trainEnd+1:valEnd);
        testIndices = indices(valEnd+1:end);
        
        % Add to appropriate datasets
        XTrain = [XTrain, X(trainIndices)];
        YTrain = [YTrain, Y(trainIndices)];
        
        XVal = [XVal, X(valIndices)];
        YVal = [YVal, Y(valIndices)];
        
        XTest = [XTest, X(testIndices)];
        YTest = [YTest, Y(testIndices)];
    end
end

% Finalize the datasets
% Shuffle the train set again to mix different simulations
trainIndices = randperm(length(XTrain));
XTrain = XTrain(trainIndices);
YTrain = YTrain(trainIndices);

% % Concatenate outputs for training with networks
YTrain = cat(2, YTrain{:})';
YVal = cat(2, YVal{:})';
YTest = cat(2, YTest{:})';

fprintf("Training: %d \nValidation: %d \nTest: %d\n", length(XTrain), length(XVal), length(XTest));

%% Physics-Informed Neural Network Training
training_start_time = tic; % Start the timer

% Calculate normalization parameters using min-max instead of z-score
fprintf("\n=== Normalization Parameters ===\n");
featureMin = min(cat(1, my_in));
featureMax = max(cat(1, my_in));

% For outputs (first 6 columns)
outputMin = featureMin(1:6);  % First 6 features match outputs 
outputMax = featureMax(1:6);  % Exclude the disturbance feature

% Show normalization parameters
fprintf("Feature mins: [%s]\n", sprintf('%.4f ', featureMin));
fprintf("Feature maxs: [%s]\n", sprintf('%.4f ', featureMax));
fprintf("Output mins: [%s]\n", sprintf('%.4f ', outputMin));
fprintf("Output maxs: [%s]\n", sprintf('%.4f ', outputMax));

% Store them for use in loss calculation
normParams = struct();
normParams.featureMin = featureMin;
normParams.featureMax = featureMax;
normParams.outputMin = outputMin;
normParams.outputMax = outputMax;

% Define scaling range
normParams.minVal = -1;  % Scale to [-1, 1] range
normParams.maxVal = 1;

% Prevent division by zero in normalization
featureRange = featureMax - featureMin;
featureRange(featureRange < 1e-6) = 1e-6;
normParams.featureRange = featureRange;

outputRange = outputMax - outputMin;
outputRange(outputRange < 1e-6) = 1e-6;
normParams.outputRange = outputRange;

% Network architecture
layers = [ ...
    sequenceInputLayer(7)
    bilstmLayer(128, 'OutputMode', 'sequence')
    dropoutLayer(0.3)
    bilstmLayer(64, 'OutputMode', 'sequence')
    dropoutLayer(0.3)
    fullyConnectedLayer(32, 'WeightL2Factor', 0.001)
    dropoutLayer(0.3)
    fullyConnectedLayer(6)
    ];

% Initialize network
net = dlnetwork(layers);

% Move network to GPU if available
if useGPU
    try
        % Convert learnable parameters to gpuArrays correctly
        for i = 1:size(net.Learnables, 1)
            net.Learnables.Value{i} = gpuArray(net.Learnables.Value{i});
        end
        fprintf('Network configured for GPU execution\n');
    catch gpuErr
        warning('GPU error occurred: %s. Reverting to CPU.', gpuErr.message);
        useGPU = false;
    end
end

% Training parameters
numEpochs = 1000;
miniBatchSize = 128;
initialLR = 0.001;   
learningRate = initialLR;

% Early stopping parameters
bestValLoss = Inf;
patience = 20;
patienceCounter = 0;
bestNet = [];  % Store best network

% Initialize Adam optimizer state
mp = [];
vp = [];

% Initialize adaptive weights
lambda_pred_max = 3.0;  % Maximum prediction weight 2
lambda_phys_min = 0.1;  % Minimum physics weight 0.1
lambda_phys_max = 10.0;  % Maximum physics weight 3

fprintf("\nLoss weights: initial pred %.1f→%.1f, physics %.1f→%.1f\n", ...
    lambda_pred_max, lambda_pred_max, lambda_phys_min, lambda_phys_max);

% Initialize metrics tracking
metrics = struct('Epoch', [], 'Iteration', [], 'TrainLoss', [], ...
                'PredLoss', [], 'PhysLoss', [], 'ValLoss', [], ...
                'LearningRate', [], 'Time', [], 'TrainAccuracy', [], 'ValAccuracy', []);

% Create and configure the training figure
fig = figure('Name', 'Training Progress Monitoring', 'NumberTitle', 'off');
set(fig, 'WindowState', 'maximized'); % Maximize the window
set(fig, 'Color', 'white'); % White background for better visibility

% Create a better layout for subplots (updated for 6 plots)
subplot_positions = [
    0.08 0.55 0.30 0.35;  % Top left - Loss
    0.42 0.55 0.30 0.35;  % Top middle - Loss Components
    0.76 0.55 0.20 0.35;  % Top right - Learning Rate
    0.08 0.10 0.30 0.35;  % Bottom left - Accuracy
    0.42 0.10 0.30 0.35;  % Bottom middle - Empty for now
    0.76 0.10 0.20 0.35   % Bottom right - Info Panel
];

% Create subplots with custom positions
subplot('Position', subplot_positions(1,:));
lineLossTrain = animatedline('Color',[0 0.7 0.3], 'LineWidth', 1.5);
lineLossVal = animatedline('Color',[0 0.3 0.7], 'LineWidth', 1.5);
title('Training vs Validation Loss', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Iteration', 'FontSize', 10);
ylabel('Loss', 'FontSize', 10);
legend({'Training','Validation'}, 'Location', 'northeast');
grid on;
ax1 = gca;
ax1.XMinorGrid = 'on';
ax1.YMinorGrid = 'on';

subplot('Position', subplot_positions(2,:));
lineLossComponents = animatedline('Color',[0.9 0.3 0.1], 'LineWidth', 1.5);
lineLossPhysics = animatedline('Color',[0.1 0.3 0.9], 'LineWidth', 1.5);
title('Loss Components', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Iteration', 'FontSize', 10);
ylabel('Loss Value', 'FontSize', 10);
legend({'Prediction','Physics'}, 'Location', 'northeast');
grid on;
ax2 = gca;
ax2.XMinorGrid = 'on';
ax2.YMinorGrid = 'on';

subplot('Position', subplot_positions(3,:));
lineLR = animatedline('Color',[0.5 0.2 0.9], 'LineWidth', 1.5);
title('Learning Rate Schedule', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Epoch', 'FontSize', 10);
ylabel('Learning Rate', 'FontSize', 10);
grid on;
ax3 = gca;
ax3.XMinorGrid = 'on';
ax3.YMinorGrid = 'on';

% Add new subplot for Accuracy
subplot('Position', subplot_positions(4,:));
lineAccTrain = animatedline('Color',[0 0.7 0.3], 'LineWidth', 1.5);
lineAccVal = animatedline('Color',[0 0.3 0.7], 'LineWidth', 1.5);
title('Training vs Validation Accuracy (R²)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Iteration', 'FontSize', 10);
ylabel('R² Value', 'FontSize', 10);
legend({'Training','Validation'}, 'Location', 'northeast');
ylim([0 1]);  % R² is typically between 0 and 1
grid on;
ax4 = gca;
ax4.XMinorGrid = 'on';
ax4.YMinorGrid = 'on';

% Add reserved space for additional metrics in the future
subplot('Position', subplot_positions(5,:));
title('Reserved for Additional Metrics', 'FontSize', 12, 'FontWeight', 'bold');
text(0.5, 0.5, 'This space is reserved for future metrics', 'FontSize', 10, ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
axis off;

% Create a panel for training info with well-defined borders
info_panel = uipanel('Position', subplot_positions(6,:), ...
                    'Title', 'Training Progress', ...
                    'FontSize', 12, ...
                    'FontWeight', 'bold', ...
                    'BackgroundColor', [0.97 0.97 1.0]);

% Add text inside the panel with proper positioning
text_info = uicontrol('Parent', info_panel, ...
                     'Style', 'text', ...
                     'Units', 'normalized', ...
                     'Position', [0.05 0.05 0.9 0.9], ...
                     'String', 'Initializing training...', ...
                     'FontSize', 10, ...
                     'HorizontalAlignment', 'left', ...
                     'BackgroundColor', [0.97 0.97 1.0]);

% Function to calculate R² (coefficient of determination) for regression accuracy
function r_squared = calculateRSquared(predictions, targets)
    % Move to CPU if necessary
    if isa(predictions, 'gpuArray')
        predictions = gather(predictions);
    end
    if isa(targets, 'gpuArray')
        targets = gather(targets);
    end
    
    % Extract data if dlarray
    if isa(predictions, 'dlarray')
        predictions = extractdata(predictions);
    end
    if isa(targets, 'dlarray')
        targets = extractdata(targets);
    end
    
    % Calculate R² (coefficient of determination)
    SST = sum((targets - mean(targets, 'all')).^2, 'all');
    SSR = sum((targets - predictions).^2, 'all');
    
    % Prevent division by zero
    if SST < 1e-10
        r_squared = 0;  % If no variance in targets, R² is meaningless
    else
        r_squared = 1 - (SSR / SST);
        
        % Clamp to [0, 1] range - negative R² values are possible but not helpful for visualization
        r_squared = max(0, min(1, r_squared));
    end
end

% Helper function for normalizing sequence data
function XBatchDL = normalizeAndPrepareSequence(XBatch, normParams, useGPU)
    XBatchDL = {};
    for j = 1:length(XBatch)
        % Apply min-max normalization
        normalized_data = normParams.minVal + ...
            (XBatch{j} - normParams.featureMin) ./ normParams.featureRange * ...
            (normParams.maxVal - normParams.minVal);
            
        % Convert to dlarray and move to GPU if available
        if useGPU
            normalized_data = gpuArray(normalized_data);
        end
        XBatchDL{j} = dlarray(normalized_data', 'CBT');  % [7 × sequence_length × 1]
    end
end

% Helper function for calculating estimated time remaining
function remainingTimeStr = estimateTimeRemaining(elapsedTime, currentIteration, totalIterations)
    if currentIteration < 10 % Avoid division by small numbers
        remainingTimeStr = 'Calculating...';
    else
        iterationsRemaining = totalIterations - currentIteration;
        timePerIteration = elapsedTime / currentIteration;
        remainingTime = timePerIteration * iterationsRemaining;
        remainingTimeStr = formatElapsedTime(remainingTime);
    end
end

iteration = 0;
totalIterations = numEpochs * ceil(length(XTrain)/miniBatchSize);

% Add initial point to learning rate plot
addpoints(lineLR, 0, learningRate);

for epoch = 1:numEpochs
    % Shuffle training data
    idx = randperm(length(XTrain));
    XTrainShuffled = XTrain(idx);
    YTrainShuffled = YTrain(idx,:);
    
    % Calculate adaptive weights based on training progress
    epoch_fraction = min(epoch / 200, 1); % Normalized training progress (cap at 1)
    lambda_pred = lambda_pred_max; % Keep prediction weight constant
    lambda_phys = lambda_phys_min + (lambda_phys_max - lambda_phys_min) * epoch_fraction; % adjust to enable pinn
    
    % Learning rate scheduling - simplified
    if mod(epoch, 50) == 0 && epoch > 0
        learningRate = initialLR * 0.7^(epoch/50);
        fprintf('Epoch %d: Learning rate decreased to %.6f\n', epoch, learningRate);
        
        % Update learning rate plot
        addpoints(lineLR, epoch, learningRate);
    end
    
    % Update progress information string
    elapsed = toc(training_start_time);
    remainingTime = estimateTimeRemaining(elapsed, iteration, totalIterations);
    
    infoString = sprintf(['Epoch: %d/%d\n' ...
                         'Learning Rate: %.6f\n' ...
                         'lambda Pred: %.2f\n' ...
                         'lambda Phys: %.2f\n\n' ...
                         'Best Val Loss: %.6f\n' ...
                         'Patience: %d/%d\n\n' ...
                         'Elapsed: %s\n' ...
                         'Remaining: %s'], ...
                         epoch, numEpochs, learningRate, lambda_pred, lambda_phys, ...
                         bestValLoss, patienceCounter, patience, ...
                         formatElapsedTime(elapsed), remainingTime);
    text_info.String = infoString;
    
    if mod(epoch, 10) == 0
        fprintf('Epoch %d: Using lambda_pred=%.2f, lambda_phys=%.2f, lr=%.6f\n', ...
            epoch, lambda_pred, lambda_phys, learningRate);
    end

    % Mini-batch training
    for i = 1:miniBatchSize:length(XTrain)
        iteration = iteration + 1;
        
        % Get mini-batch
        batchEnd = min(i + miniBatchSize - 1, length(XTrain));
        XBatch = XTrainShuffled(i:batchEnd);
        YBatch = YTrainShuffled(i:batchEnd, :);
        
        % Format data for network using helper function
        XBatchDL = normalizeAndPrepareSequence(XBatch, normParams, useGPU);
        
        % Normalize targets using min-max and convert to GPU if needed
        YBatch_normalized = YBatch';  % [6 × batchSize]
        if useGPU
            YBatch_normalized = gpuArray(YBatch_normalized);
        end
        YBatchDL = dlarray(YBatch_normalized, 'CB');  % [6 × batchSize]
        
        % Compute loss and gradients with dlfeval
        [loss, gradients, lossPred, lossPhys, predictions] = dlfeval(@modelLoss_local, net, XBatchDL, YBatchDL, lambda_pred, lambda_phys, m, k, kp, d, dt, normParams);
        
        % Calculate accuracy (R²) for this batch
        r_squared = calculateRSquared(predictions, YBatch'); 
        
        % Update network using Adam
        [net, mp, vp] = adamupdate(net, gradients, mp, vp, iteration, learningRate);
        
        % Update plot - safely handle extractdata and GPU arrays
        lossValue = double(gather(extractdata(loss)));
        lossPredValue = double(gather(extractdata(lossPred)));
        lossPhysValue = double(gather(extractdata(lossPhys)));
        
        % Store metrics
        metrics.Iteration(end+1) = iteration;
        metrics.TrainLoss(end+1) = lossValue;
        metrics.PredLoss(end+1) = lossPredValue;
        metrics.PhysLoss(end+1) = lossPhysValue;
        metrics.TrainAccuracy(end+1) = r_squared;
        metrics.Time(end+1) = toc(training_start_time);
        
        % Update plots
        addpoints(lineLossTrain, iteration, lossValue);
        addpoints(lineLossComponents, iteration, lossPredValue);
        addpoints(lineLossPhysics, iteration, lossPhysValue);
        addpoints(lineAccTrain, iteration, r_squared);
        
        if mod(iteration, 50) == 0
            fprintf('Epoch %d, Iteration %d: Loss = %.6f (Pred: %.6f, Phys: %.6f), R² = %.4f\n', ...
                epoch, iteration, lossValue, lossPredValue, lossPhysValue, r_squared);
                
            % Update progress information with latest metrics
            infoString = sprintf(['Epoch: %d/%d\n' ...
                                 'Iteration: %d/%d\n' ...
                                 'Learning Rate: %.6f\n' ...
                                 'lambda Pred: %.2f\n' ...
                                 'lambda Phys: %.2f\n\n' ...
                                 'Train Loss: %.6f\n' ...
                                 'Train R²: %.4f\n' ...
                                 'Best Val Loss: %.6f\n' ...
                                 'Patience: %d/%d\n\n' ...
                                 'Elapsed: %s\n' ...
                                 'Remaining: %s'], ...
                                 epoch, numEpochs, iteration, totalIterations, ...
                                 learningRate, lambda_pred, lambda_phys, lossValue, ...
                                 r_squared, bestValLoss, patienceCounter, patience, formatElapsedTime(elapsed), ...
                                 estimateTimeRemaining(elapsed, iteration, totalIterations));
            text_info.String = infoString;
        end
        
        drawnow;
    end
    
    % Validation every 10 epochs
    if mod(epoch, 10) == 0
        fprintf('Running validation...\n');
        valLoss = 0;
        valAccuracy = 0;
        numValBatches = 0;
        
        % Store all validation predictions and targets for accurate R² calculation
        all_val_preds = [];
        all_val_targets = [];
        
        for j = 1:length(XVal)
            % Normalize validation data using min-max
            normalized_val = normParams.minVal + ...
                (XVal{j} - normParams.featureMin) ./ normParams.featureRange * ...
                (normParams.maxVal - normParams.minVal);
                
            % Move to GPU if available
            if useGPU
                normalized_val = gpuArray(normalized_val);
            end
            
            XValDL = dlarray(normalized_val', 'CBT');
            
            % Prepare Y validation data
            YVal_normalized = YVal(j,:)';
            if useGPU
                YVal_normalized = gpuArray(YVal_normalized);
            end
            
            YValDL = dlarray(YVal_normalized, 'CB');
            
            % Use dlfeval with the VALIDATION function (no gradients)
            [lossVal, predictions] = dlfeval(@validationLoss, net, {XValDL}, YValDL, lambda_pred, lambda_phys, m, k, kp, d, dt, normParams);
            
            % Collect predictions and targets for overall R² calculation
            all_val_preds = [all_val_preds, gather(extractdata(predictions))];
            all_val_targets = [all_val_targets, YVal(j,:)'];
            
            % Safe extraction for accumulation
            valLoss = valLoss + double(gather(extractdata(lossVal)));
            numValBatches = numValBatches + 1;
        end
        
        valLoss = valLoss / numValBatches;
        
        % Calculate validation accuracy (R²) for all validation data
        valAccuracy = calculateRSquared(all_val_preds, all_val_targets);
        
        % Store validation metrics
        metrics.Epoch(end+1) = epoch;
        metrics.ValLoss(end+1) = valLoss;
        metrics.ValAccuracy(end+1) = valAccuracy;
        metrics.LearningRate(end+1) = learningRate;
        
        % Update validation plots
        addpoints(lineLossVal, iteration, valLoss);
        addpoints(lineAccVal, iteration, valAccuracy);
        
        fprintf('Validation Loss: %.6f, R²: %.4f\n', valLoss, valAccuracy);
        
        % Early stopping check
        if valLoss < bestValLoss
            bestValLoss = valLoss;
            bestNet = net;  % Save the best network
            patienceCounter = 0;
            fprintf('New best validation loss: %.6f\n', bestValLoss);
        else
            patienceCounter = patienceCounter + 1;
            fprintf('Patience counter: %d/%d\n', patienceCounter, patience);
            if patienceCounter >= patience
                fprintf('Early stopping at epoch %d\n', epoch);
                break;
            end
        end
        
        % Update progress information
        infoString = sprintf(['Epoch: %d/%d\n' ...
                             'Iteration: %d/%d\n' ...
                             'Learning Rate: %.6f\n' ...
                             'Lambda Pred: %.2f\n' ...
                             'Lambda Phys: %.2f\n\n' ...
                             'Train Loss: %.6f\n' ...
                             'Train R²: %.4f\n' ...
                             'Val Loss: %.6f\n' ...
                             'Val R²: %.4f\n' ...
                             'Best Val Loss: %.6f\n' ...
                             'Patience: %d/%d\n\n' ...
                             'Elapsed: %s\n' ...
                             'Remaining: %s'], ...
                             epoch, numEpochs, iteration, totalIterations, ...
                             learningRate, lambda_pred, lambda_phys, ...
                             lossValue, metrics.TrainAccuracy(end), valLoss, valAccuracy, ...
                             bestValLoss, patienceCounter, patience, formatElapsedTime(elapsed), ...
                             estimateTimeRemaining(elapsed, iteration, totalIterations));
        text_info.String = infoString;
        
        drawnow;
    end
end

% If we have a best network from early stopping, use it
if ~isempty(bestNet)
    fprintf('Using best network from early stopping (validation loss: %.6f)\n', bestValLoss);
    net = bestNet;
end

% Update the figure title to show completion
fig.Name = sprintf('Training Progress Monitoring(Complete) - Final Val Loss: %.6f, Val R²: %.4f', bestValLoss, valAccuracy);

% Save the training progress figure
savefig(fig, 'training_progress.fig');
fprintf('Training progress figure saved as training_progress.fig\n');

% Save training metrics for future analysis
save('training_metrics.mat', 'metrics');
fprintf('Training metrics saved as training_metrics.mat\n');

% Move network back to CPU for saving if needed
if useGPU
    try
        for i = 1:size(net.Learnables, 1)
            net.Learnables.Value{i} = gather(net.Learnables.Value{i});
        end
    catch
        warning('Error gathering network parameters from GPU. Network might not be correctly transferred to CPU.');
    end
end

%% Save Best Network
% Generate a unique filename that doesn't already exist
fileExists = true;
while fileExists
    % Create random characters (mix of digits and lowercase letters)
    chars = zeros(1, 4);
    for i = 1:4
        if rand < 0.5
            % Generate a random digit (ASCII 48-57)
            chars(i) = randi([48, 57]);
        else
            % Generate a random lowercase letter (ASCII 97-122)
            chars(i) = randi([97, 122]);
        end
    end
    
    % Create the filename
    filename = ['bestnet_', char(chars), '.mat'];
    
    % Check if the file already exists
    fileExists = exist(filename, 'file') == 2;
    
    if fileExists
        fprintf('File %s already exists, generating a new name...\n', filename);
    else
        fprintf('Generated unique filename: %s\n', filename);
    end
end

% Now you can use the filename for saving
save(filename, 'net', 'normParams');

%% Visual Test
seq = 1;%length(XTest);
% Format the input data for prediction with min-max normalization
normalized_test = normParams.minVal + ...
 (XTest{1,seq} - normParams.featureMin) ./ normParams.featureRange * ...
 (normParams.maxVal - normParams.minVal);

% Move to GPU for prediction if available
if useGPU
    normalized_test = gpuArray(normalized_test);
end

XTest_formatted = dlarray(normalized_test', 'CBT');
YPred_normalized = forward(net, XTest_formatted);

% % Extract the final prediction
% if ndims(YPred_normalized) == 3
%  YPred_normalized = YPred_normalized(:,1,end);
% end

% Move prediction back to CPU for visualization
YPred_normalized = gather(extractdata(YPred_normalized));

% Denormalize predictions using min-max
YPred = (YPred_normalized - normParams.minVal) ./ (normParams.maxVal - normParams.minVal) .* ...
 normParams.outputRange' + normParams.outputMin';

fig = figure('Name', 'Visual Test', 'NumberTitle', 'off');
% Create figure with enough space for legend at bottom
set(gcf, 'Position', [100, 100, 800, 600]);

% Initialize arrays to store plot handles for the legend
predictionHandles = gobjects(1,6);
truthHandles = gobjects(1,6);
sequenceHandles = gobjects(1,6);

% Subplot for x1
subplot(2,3,1)
predictionHandles(1) = scatter(length(XTest{1,seq})+1, YPred(1), "filled");
hold on;
truthHandles(1) = scatter(length(XTest{1,seq})+1, YTest(seq,1), "MarkerFaceColor", "r", "MarkerEdgeColor", "k");
hold on;
sequenceHandles(1) = plot(XTest{1,seq}(:,1));
hold on;
ylabel("x1 (m)")
xlabel("timestep (k)")

% Subplot for x2
subplot(2,3,2)
predictionHandles(2) = scatter(length(XTest{1,seq})+1, YPred(2), "filled");
hold on;
truthHandles(2) = scatter(length(XTest{1,seq})+1, YTest(seq,2), "MarkerFaceColor", "r", "MarkerEdgeColor", "k");
hold on;
sequenceHandles(2) = plot(XTest{1,seq}(:,2));
hold on;
ylabel("x2 (m)")
xlabel("timestep (k)")
% Remove individual legend

% Subplot for x3
subplot(2,3,3)
predictionHandles(3) = scatter(length(XTest{1,seq})+1, YPred(3), "filled");
hold on;
truthHandles(3) = scatter(length(XTest{1,seq})+1, YTest(seq,3), "MarkerFaceColor", "r", "MarkerEdgeColor", "k");
hold on;
sequenceHandles(3) = plot(XTest{1,seq}(:,3));
hold on;
ylabel("x3 (m)")
xlabel("timestep (k)")

% Subplot for v1
subplot(2,3,4)
predictionHandles(4) = scatter(length(XTest{1,seq})+1, YPred(4), "filled");
hold on;
truthHandles(4) = scatter(length(XTest{1,seq})+1, YTest(seq,4), "MarkerFaceColor", "r", "MarkerEdgeColor", "k");
hold on;
sequenceHandles(4) = plot(XTest{1,seq}(:,4));
hold on;
ylabel("v1 (m/s)")
xlabel("timestep (k)")

% Subplot for v2
subplot(2,3,5)
predictionHandles(5) = scatter(length(XTest{1,seq})+1, YPred(5), "filled");
hold on;
truthHandles(5) = scatter(length(XTest{1,seq})+1, YTest(seq,5), "MarkerFaceColor", "r", "MarkerEdgeColor", "k");
hold on;
sequenceHandles(5) = plot(XTest{1,seq}(:,5));
hold on;
ylabel("v2 (m/s)")
xlabel("timestep (k)")

% Subplot for v3
subplot(2,3,6)
predictionHandles(6) = scatter(length(XTest{1,seq})+1, YPred(6), "filled");
hold on;
truthHandles(6) = scatter(length(XTest{1,seq})+1, YTest(seq,6), "MarkerFaceColor", "r", "MarkerEdgeColor", "k");
hold on;
sequenceHandles(6) = plot(XTest{1,seq}(:,6));
hold on;
ylabel("v3 (m/s)")
xlabel("timestep (k)")

% Create a single legend for all subplots
% Use the first handles for the legend
h_legend = [predictionHandles(1), truthHandles(1), sequenceHandles(1)];
leg = legend(h_legend, 'Prediction', 'Ground Truth', 'Sequence');

% Place the legend at the bottom of the figure
leg.Position = [0.35, 0.02, 0.3, 0.05];  % [x, y, width, height]
leg.Orientation = 'horizontal';
leg.Box = 'off';  % Optional: removes the box around the legend

% Adjust subplot spacing to make room for the legend
subplot_pos = get(gcf, 'DefaultAxesPosition');
set(gcf, 'DefaultAxesPosition', [subplot_pos(1), subplot_pos(2)+0.05, subplot_pos(3), subplot_pos(4)-0.05]);

% Print GPU performance information if used
if useGPU
    fprintf('\nGPU Training Performance:\n');
    fprintf('Total training time: %s\n', formatElapsedTime(toc(training_start_time)));
    fprintf('GPU Memory Used: %.2f GB\n', (gpu.TotalMemory - gpu.AvailableMemory)/1e9);
    reset(gpu);  % Release GPU memory
end

%% Local Functions
function [loss, gradients, lossPred, lossPhys, YPred] = modelLoss_local(net, XBatch, YTarget, lambda_pred, lambda_phys, m, k, kp, d, dt, normParams)
    % Initialize predictions with correct dimension
    numSequences = length(XBatch);
    YPred_normalized = zeros(6, numSequences, 'like', YTarget);
    
    % Process each sequence
    for i = 1:numSequences
        % Forward pass through network
        pred = forward(net, XBatch{i});
        
        % Extract only the final prediction from each sequence
        if ndims(pred) == 3
            % For "sequence" output mode: [features × batch × timesteps]
            YPred_normalized(:,i) = pred(:,1,end);
        elseif ndims(pred) == 2 && size(pred,2) > 1
            % For "sequence" output mode without batch dim: [features × timesteps]
            YPred_normalized(:,i) = pred(:,end);
        else
            % For "last" output mode: [features × 1]
            YPred_normalized(:,i) = pred;
        end
    end
    
    % Prediction loss in NORMALIZED space (MSE)
    % YTarget is in original units, so we need to normalize it using min-max
    YTarget_normalized = normParams.minVal + ...
        (YTarget - normParams.outputMin') ./ normParams.outputRange' * ...
        (normParams.maxVal - normParams.minVal);
    lossPred = mean((YPred_normalized - YTarget_normalized).^2, 'all');
    
    % Get the last states from input for proper physics calculations
    lastStates = zeros(7, numSequences, 'like', YTarget);
    for i = 1:numSequences
        % Extract the last timestep of each input sequence
        lastStates(:,i) = extractdata(XBatch{i}(:,end));
    end
    
    % Denormalize the last states to get physical units for disturbance
    lastStates_denorm = (lastStates - normParams.minVal) ./ (normParams.maxVal - normParams.minVal) .* ...
        normParams.featureRange' + normParams.featureMin';
    
    % Denormalize predictions for PHYSICS calculations
    % This ensures physics calculations are done with actual physical units
    YPred = (YPred_normalized - normParams.minVal) ./ (normParams.maxVal - normParams.minVal) .* ...
        normParams.outputRange' + normParams.outputMin';
    
    % Physics loss using DENORMALIZED values (proper physical units)
    lossPhys = physicsLoss(YPred, YTarget, m, k, kp, d, dt);
    
    % Total loss with weighting - keep as dlarray
    loss = lambda_pred * lossPred + lambda_phys * lossPhys;
    
    % Compute gradients - this line needs dlfeval
    gradients = dlgradient(loss, net.Learnables);
end

function [loss, YPred] = validationLoss(net, XBatch, YTarget, lambda_pred, lambda_phys, m, k, kp, d, dt, normParams)
    % This is a simpler function with NO gradients for validation
    numSequences = length(XBatch);
    YPred_normalized = zeros(6, numSequences, 'like', YTarget);
    
    % Process each sequence
    for i = 1:numSequences
        % Forward pass through network
        pred = forward(net, XBatch{i});
        
        % Extract only the final prediction from each sequence
        if ndims(pred) == 3
            % For "sequence" output mode: [features × batch × timesteps]
            YPred_normalized(:,i) = pred(:,1,end);
        elseif ndims(pred) == 2 && size(pred,2) > 1
            % For "sequence" output mode without batch dim: [features × timesteps]
            YPred_normalized(:,i) = pred(:,end);
        else
            % For "last" output mode: [features × 1]
            YPred_normalized(:,i) = pred;
        end
    end
    
    % Prediction loss in normalized space (min-max)
    YTarget_normalized = normParams.minVal + ...
        (YTarget - normParams.outputMin') ./ normParams.outputRange' * ...
        (normParams.maxVal - normParams.minVal);
    lossPred = mean((YPred_normalized - YTarget_normalized).^2, 'all');
    
    % Denormalize predictions for physics calculations
    YPred = (YPred_normalized - normParams.minVal) ./ (normParams.maxVal - normParams.minVal) .* ...
        normParams.outputRange' + normParams.outputMin';
    
    % Denormalize targets
    YTarget_denorm = YTarget;  % Already in physical units
    
    % Physics loss
    lossPhys = physicsLoss(YPred, YTarget_denorm, m, k, kp, d, dt);
    
    % Total loss with weighting (NO GRADIENT COMPUTATION)
    loss = lambda_pred * lossPred + lambda_phys * lossPhys;
end

function physLoss = physicsLoss(YPred, YTarget, m, k, kp, d, dt)
    % Extract positions and velocities
    x_pred = YPred(1:3, :);      % [x1, x2, x3]
    xdot_pred = YPred(4:6, :);   % [xdot1, xdot2, xdot3]
    
    % Calculate forces based on positions and physical parameters
    f1 = k*(-2*x_pred(1,:) + x_pred(2,:)) + kp*(-x_pred(1,:).^3 + (x_pred(2,:)-x_pred(1,:)).^3);
    f2 = k*(x_pred(1,:) - 2*x_pred(2,:) + x_pred(3,:)) + kp*((x_pred(3,:)-x_pred(2,:)).^3 - (x_pred(2,:)-x_pred(1,:)).^3);
    f3 = k*(x_pred(2,:) - x_pred(3,:)) + kp*(-(x_pred(3,:)-x_pred(2,:)).^3);
    
    % Calculate expected accelerations based on forces (F = ma)
    expected_x1ddot = f1/m;
    expected_x2ddot = f2/m;
    expected_x3ddot = f3/m;
    
    % Energy-based physics loss (improved)
    E_pred = computeEnergy_local(x_pred, xdot_pred, m, k, kp);
    damping_work = d * sum(xdot_pred.^2, 1) * dt;
    energy_loss = mean(abs(E_pred - damping_work), 'all');
    
    % Newton's 2nd Law based physics loss (F = ma)
    newton_loss = mean(abs(expected_x1ddot).^2 + abs(expected_x2ddot).^2 + abs(expected_x3ddot).^2, 'all');
    
    % Combined physics loss - energy conservation and Newton's laws
    physLoss = energy_loss + 0.1 * newton_loss;
end

function E = computeEnergy_local(x, xdot, m, k, kp)
    % Kinetic energy
    KE = 0.5 * m * sum(xdot.^2, 1);
    
    % Linear potential energy
    PE_linear = 0.5 * k * (x(1,:).^2 + (x(2,:)-x(1,:)).^2 + (x(3,:)-x(2,:)).^2);
    
    % Nonlinear potential energy
    PE_nonlinear = 0.25 * kp * (x(1,:).^4 + (x(2,:)-x(1,:)).^4 + (x(3,:)-x(2,:)).^4);
    
    % Total energy
    E = KE + PE_linear + PE_nonlinear;
end

% Function to format elapsed time in HH:MM:SS format
function timeString = formatElapsedTime(elapsedSeconds)
    hours = floor(elapsedSeconds / 3600);
    minutes = floor(mod(elapsedSeconds, 3600) / 60);
    seconds = mod(elapsedSeconds, 60);
    timeString = sprintf('%02d:%02d:%02d', hours, minutes, round(seconds));
end

%% Animate MSD
% animate_msd(my_in);