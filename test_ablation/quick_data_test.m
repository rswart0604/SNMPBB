% load("ORL.mat");
% r = 40;
% X = data;
% gnd = label;
% k = 3;
% plot_name = "orl_mean";

% load("Reuters21578.mat");
load("TDT2_all.mat");
r = 20;
X = fea;
[clusterLabels, ~, idx] = unique(gnd);
% get the top 20 clusters for these two datasets
clusterSizes = accumarray(idx, 1);
[~, sortIdx] = sort(clusterSizes, 'descend');
top20_labels = clusterLabels(sortIdx(1:20));
mask = ismember(gnd, top20_labels);
X = X(mask, :);
% X = bsxfun(@rdivide, X, sqrt(sum(X.^2,2)) + eps);
% [U,S,~] = svds(X, r);
% A_dense = U * S * U';
% X = max(X, 0);
gnd = gnd(mask);
% [~, ~, gnd] = unique(gnd2


% load("COIL20.mat")
% r = 20;
% gnd = Y;
% k = 12;
% plot_name = "coil20_mean";

% load("Isolet1.mat")
% r = 26;
% X = fea;
% k = 40;
% plot_name = "isolet1_mean";

% load("2k2k.mat")
% X = fea;
% r = 10;
% k = 42;
% plot_name = "2k2k_mean";

num_runs = 3;
plot_time_limit = 10; % Show only first n seconds (adjust as needed)

% Define methods with their algorithm names for symnmf_cluster
% To add a new method, just add a new entry here
method_configs = {
    % name,            alg_param,           plot_label,  plot_color
    % 'graphsnmpbb',    'graph_snmpbb',      'Graph SNMPBB',    '#0072BD';
    % 'graphsnmpbb',    'graph_snmpbb',      'Plain',    '#0072BD', struct();
    'graphsnmpbb_modified',    'modified_graph_snmpbb',      'Plain',    '#0072BD', struct('bb', true, 'second_descent_step', true, 'nonmonotone', true);
    'graphsnmpbb_modified1',    'modified_graph_snmpbb',      'No nonmonotone',    '#D95319', struct('bb', true, 'second_descent_step', true, 'nonmonotone', false);
    'graphsnmpbb_modified2',    'modified_graph_snmpbb',      'No bb',    '#77AC30', struct('bb',false, 'second_descent_step', true, 'nonmonotone', true);
    'graphsnmpbb_modified3',    'modified_graph_snmpbb',      'Nothing',    '#8516D1', struct('bb',false, 'nonmonotone', false, 'second_descent_step', false);
    % 'anls',           'anls',              'ANLS',            '#8516D1', struct();
    % 'pgd',            'pgd',               'PGD',             '#77AC30', struct();
    % 'modified_pgd',   'modified_pgd',      'PGD modified',    '#8500FF', struct();
};

num_methods = size(method_configs, 1);
method_names = method_configs(:, 1);

% Initialize storage using struct array
results = struct();
for i = 1:num_methods
    method = method_names{i};
    results.(method).time = cell(num_runs, 1);
    results.(method).acc_raw = cell(num_runs, 1);
    results.(method).relres_raw = cell(num_runs, 1);
end

[m,~] = size(X);

for x = 1:num_runs
    tic;
    W0 = 2 * full(sqrt(mean(mean(X)) / r)) * rand(m, r);
    H0 = 2 * full(sqrt(mean(mean(X)) / r)) * rand(r,m);    
    params.truelabel = gnd;
    params.Hinit = W0;
    params.Winit = H0;
    
    % Loop through all methods
    for i = 1:num_methods
        method_name = method_names{i};
        alg_param = method_configs{i, 2}
        
        % Set algorithm parameter
        params.alg = alg_param;

        if strcmp(params.alg, 'modified_graph_snmpbb')
            mystruct = method_configs{i,5};
            fields = fieldnames(mystruct);
            for j = 1:numel(fields)
                fieldname = fields{j};
                params.(fieldname) = mystruct.(fieldname);
            end
        end
            
        
        % Call symnmf_cluster
        [~, output, acc] = symnmf_cluster(X, r, params);
        
        % Store raw data after cleaning
        [results.(method_name).time{x}, results.(method_name).acc_raw{x}] = ...
            clean_time_acc(output.total_time(:), acc(:));
        [results.(method_name).relres_time{x}, results.(method_name).relres_raw{x}] = ...
    clean_time_acc(output.total_time(:), output.relres(:));

    end
    
    toc;
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

% Add this parameter to control plot time range (in seconds)

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

% Optional: Set x-axis limits explicitly
if plot_time_limit > 0
    xlim([0, plot_time_limit]);
end

% Export data
export_data = t_plot';
for i = 1:num_methods
    method = method_names{i};
    export_data = [export_data, mean_data.(method)'];
end
% writematrix(export_data, 'data/reuters.csv');


for i = 1:num_methods
    method = method_names{i};
    results.(method).relres_all = zeros(num_runs, length(t_grid));
end
for x = 1:num_runs
    for i = 1:num_methods
        method = method_names{i};
        results.(method).relres_all(x,:) = interpolate_with_clipping(...
            results.(method).relres_time{x}, ...
            results.(method).relres_raw{x}, ...
            t_grid);
    end
end

% Plot residual
figure; hold on
for i = 1:num_methods
    method = method_names{i};
    mean_relres = mean(results.(method).relres_all(:,time_idx), 1);
    plot(t_plot, mean_relres, 'Color', method_configs{i,4}, 'LineWidth', 2, 'DisplayName', method_configs{i,3});
end
ax = gca;
ax.FontSize = 14;
ax.TickLabelInterpreter = 'latex';
xlabel('$\textnormal{Time (s)}$', 'Interpreter', 'Latex', 'Fontsize', 20);
ylabel('$\textnormal{Relative Residual}$', 'Interpreter', 'Latex', 'Fontsize', 20);
lgd = legend('Location', 'best');
lgd.FontSize = 16;
lgd.Interpreter = 'latex';
if plot_time_limit > 0, xlim([0, plot_time_limit]); end



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