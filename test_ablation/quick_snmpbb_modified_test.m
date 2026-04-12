clear;clc
X = load('3000_bullseye.data'); 
r = 3;

% gnd = [ones(60,1); 2*ones(140,1); 3*ones(100,1)]; k =15;  % 300_bullseye.data
% gnd = [ones(3000,1); 2*ones(3000,1); 3*ones(3000,1)]; k = 100;  % 9000_bullseye.data
% gnd = [ones(500,1); 2*ones(500,1); 3*ones(500,1)]; k = 40;  % 1500_bullseye.data
gnd = [ones(1000,1); 2*ones(1000,1); 3*ones(1000,1)]; k = 80;  % 3000_bullseye.data
% gnd = [ones(75,1); 2*ones(75,1); 3*ones(75,1); 4*ones(75,1)]; k=40;  % four_quadrants.data

num_runs = 4;
plot_time_limit = 6;

% Define methods with their calling functions
% To add a new method, just add a new entry here
method_configs = {
    % name,           function_handle,                                                plot_label,        plot_color
    % 'graphsnmpbbog',   @(A,r,gnd,W0,H0) Graph_SNMPBB(A,r,'truelabel',gnd,'do_preprocess',false, 'W_INIT', W0, 'H_INIT', H0), 'Og',   '#D95319';
    'graphsnmpbb',   @(A,r,gnd,W0,H0) Graph_SNMPBB_modified(A,r,'truelabel',gnd,'do_preprocess',false, 'W_INIT', W0, 'H_INIT', H0), 'Everything',   '#0072BD';
    'graphsnmpbb2',   @(A,r,gnd,W0,H0) Graph_SNMPBB_modified(A,r,'truelabel',gnd,'do_preprocess',false,  'nonmonotone', false, 'W_INIT', W0, 'H_INIT', H0), 'No nonmonotone',   '#8516D1';
    'graphsnmpbb3',   @(A,r,gnd,W0,H0) Graph_SNMPBB_modified(A,r,'truelabel',gnd,'do_preprocess',false,'bb', false, 'W_INIT', W0, 'H_INIT', H0), 'No bb',   '#2FBEEF';
    'graphsnmpbb4',   @(A,r,gnd,W0,H0) Graph_SNMPBB_modified(A,r,'truelabel',gnd,'do_preprocess',false, 'bb', false, 'nonmonotone', false, 'second_descent_step', false, 'W_INIT', W0, 'H_INIT', H0), 'Nothing!',   '#77AC30';
    % 'graphsnmpbb3',   @(A,r,gnd) Graph_SNMPBB_modified(A,r,'truelabel',gnd,'do_preprocess',false), 'old',   '#8516D1';
    % 'anls',          @(A,r,gnd) symnmf_anls(A,r,struct('truelabel',gnd)),              'ANLS',           '#8516D1';
    % 'newton',        @(A,r,gnd) symnmf_newton(A,r,struct('truelabel',gnd)),            'Newton',         '#2FBEEF';
    % 'snmpbb',        @(A,r,gnd) NMF_SNMPBB(A,r,'truelabel',gnd),                       'SNMPBB',         '#D95319';
    % 'pgd',           @(A,r,gnd) PGD(A,r,'TRUELABEL',gnd),                              'PGD',            '#77AC30';
};

num_methods = size(method_configs, 1);
method_names = method_configs(:, 1);

% Initialize storage using struct array
results = struct();
for i = 1:num_methods
    method = method_names{i};
    results.(method).time = cell(num_runs, 1);
    results.(method).acc_raw = cell(num_runs, 1);
end

init_tic = tic;
V = X;
[m, ~] = size(V);
distances = pdist2(V, V, 'euclidean');

% default sigma if not provided
if ~exist('sigma','var')
    sigma = 35;
end

sigma = mean(distances(:))*40;

W = exp(- distances.^2 / (2*sigma^2));

for i = 1:m
    [~, idx] = sort(distances(i, :), 'ascend');
    neighbors = idx(2:k+1);  % ignore the first one (distance to itself)
    mask = true(1, m);
    mask(neighbors) = false;
    W(i, mask) = 0;
end

% Symmetrize affinity
A = (W + W') / 2;
init_time = toc(init_tic);

for x = 1:num_runs
    fprintf('Run %d/%d\n', x, num_runs);
    init_tic = tic;
    
    % Loop through all methods
    for i = 1:num_methods
        [m,~] = size(X);
        W0 = 2 * full(sqrt(mean(mean(A)) / r)) * rand(m, r);
        H0 = 2 * full(sqrt(mean(mean(A)) / r)) * rand(r,m);    

        method_name = method_names{i};
        disp(method_name);
        method_func = method_configs{i, 2};
        
        % Call the method function
        [~, ~, output, acc] = method_func(A, r, gnd, W0, H0);
        
        % Add initialization time to output
        output.time = output.time + init_time;
        output.total_time = output.total_time + init_time;
        
        % Store raw data after cleaning
        [results.(method_name).time{x}, results.(method_name).acc_raw{x}] = ...
            clean_time_acc(output.total_time(:), acc(:));
    end
    
    toc(init_tic)
end

% Find the maximum time across all methods and runs
max_time = 0;
for i = 1:num_methods
    method = method_names{i};
    for x = 1:num_runs
        max_time = max(max_time, max(results.(method).time{x}));
    end
end

% Use a common time grid
t_grid = linspace(0, max_time, 500);

% Storage for interpolated curves
for i = 1:num_methods
    method = method_names{i};
    results.(method).acc_all = zeros(num_runs, length(t_grid));
end

% Interpolation for each run with proper clipping
for x = 1:num_runs
    for i = 1:num_methods
        method = method_names{i};
        results.(method).acc_all(x,:) = interpolate_with_clipping(...
            results.(method).time{x}, ...
            results.(method).acc_raw{x}, ...
            t_grid);
    end
end


% Find the indices corresponding to the time limit
if plot_time_limit > 0
    time_idx = t_grid <= plot_time_limit;
    t_plot = t_grid(time_idx);
else
    % If plot_time_limit is 0 or negative, show all data
    time_idx = true(size(t_grid));
    t_plot = t_grid;
end

% Compute mean accuracy for each method
figure; hold on
mean_data = struct();
for i = 1:num_methods
    method = method_names{i};
    mean_data.(method) = mean(results.(method).acc_all(:,time_idx), 1);
end

% Plot all methods
for i = 1:num_methods
    method = method_names{i};
    plot_label = method_configs{i, 3};
    plot_color = method_configs{i, 4};
    
    plot(t_plot, mean_data.(method), 'Color', plot_color, 'LineWidth', 2, 'DisplayName', plot_label);
end

ax = gca;
ax.FontSize = 14;
ax.TickLabelInterpreter = 'latex';
xlabel('$\textnormal{Time (s)}$', 'Interpreter', 'Latex', 'Fontsize', 20);
ylabel('$\textnormal{Accuracy}$', 'Interpreter', 'Latex', 'Fontsize', 20);
lgd = legend('Location', 'best');
lgd.FontSize = 16;
lgd.Interpreter = 'latex';

% Export data
export_data = t_plot';
for i = 1:num_methods
    method = method_names{i};
    export_data = [export_data, mean_data.(method)'];
end


% Helper function to clean time/accuracy data
function [clean_time, clean_acc] = clean_time_acc(time_vec, acc_vec)
    % Remove any NaN or Inf values
    valid_idx = isfinite(time_vec) & isfinite(acc_vec);
    time_vec = time_vec(valid_idx);
    acc_vec = acc_vec(valid_idx);
    
    % Sort by time
    [time_vec, sort_idx] = sort(time_vec);
    acc_vec = acc_vec(sort_idx);
    
    % Remove duplicate time points (keep the last accuracy for each time)
    [clean_time, unique_idx] = unique(time_vec, 'last');
    clean_acc = acc_vec(unique_idx);
    
    % Clip accuracy at 1.0 and stop there
    first_perfect = find(clean_acc >= 1.0, 1, 'first');
    if ~isempty(first_perfect)
        clean_time = clean_time(1:first_perfect);
        clean_acc = clean_acc(1:first_perfect);
        clean_acc(end) = 1.0;  % Ensure it's exactly 1.0
    end
end

% Helper function to interpolate with proper clipping and zero start
function interp_acc = interpolate_with_clipping(time_vec, acc_vec, t_grid)
    if isempty(time_vec) || isempty(acc_vec)
        interp_acc = zeros(size(t_grid));
        return;
    end
    
    % Prepend t=0 with accuracy=0 if the first measurement isn't at t=0
    if time_vec(1) > 0
        time_vec = [0; time_vec(:)];
        acc_vec = [0; acc_vec(:)];
    end
    
    % Find where we should stop (when accuracy hits 1.0 or at the end of data)
    max_time = time_vec(end);
    
    % Initialize output
    interp_acc = zeros(size(t_grid));
    
    % Only interpolate up to the maximum time we have data for
    valid_t_idx = t_grid <= max_time;
    
    if any(valid_t_idx)
        % Interpolate only the valid portion (now 'linear' without 'extrap')
        interp_acc(valid_t_idx) = interp1(time_vec, acc_vec, t_grid(valid_t_idx), 'linear');
        
        % For times beyond our data, use the last known accuracy
        interp_acc(~valid_t_idx) = acc_vec(end);
        
        % Ensure accuracy stays within bounds [0, 1]
        interp_acc = max(0, min(interp_acc, 1.0));
        
        % If we hit perfect accuracy, maintain it
        if acc_vec(end) >= 1.0
            perfect_idx = t_grid >= time_vec(end);
            interp_acc(perfect_idx) = 1.0;
        end
    end
end