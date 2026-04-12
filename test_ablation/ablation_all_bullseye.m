clear; clc

% =========================================================================
% DATASET CONFIGURATIONS
% Add or remove datasets here. Each row is:
%   { 'filename', gnd_vector, k, plot_time_limit }
% =========================================================================
dataset_configs = {
    '300_bullseye.data',  [ones(60,1);  2*ones(140,1); 3*ones(100,1)], 15,  1;
    '1500_bullseye.data', [ones(500,1); 2*ones(500,1); 3*ones(500,1)], 40,  3;
    '3000_bullseye.data', [ones(1000,1);2*ones(1000,1);3*ones(1000,1)],80,  6;
    '9000_bullseye.data', [ones(3000,1);2*ones(3000,1);3*ones(3000,1)],100, 10;
};

% =========================================================================
% METHOD CONFIGURATIONS
% { name, function_handle, plot_label, plot_color }
% =========================================================================
method_configs = {
    'graphsnmpbb',  @(A,r,gnd,W0,H0) Graph_SNMPBB_modified(A,r,'truelabel',gnd,'do_preprocess',false,'W_INIT',W0,'H_INIT',H0),                                                                   'Everything',     '#0072BD';
    'graphsnmpbb2', @(A,r,gnd,W0,H0) Graph_SNMPBB_modified(A,r,'truelabel',gnd,'do_preprocess',false,'nonmonotone',false,'W_INIT',W0,'H_INIT',H0),                                               'No nonmonotone', '#8516D1';
    'graphsnmpbb3', @(A,r,gnd,W0,H0) Graph_SNMPBB_modified(A,r,'truelabel',gnd,'do_preprocess',false,'bb',false,'W_INIT',W0,'H_INIT',H0),                                                        'No bb',          '#2FBEEF';
    'graphsnmpbb4', @(A,r,gnd,W0,H0) Graph_SNMPBB_modified(A,r,'truelabel',gnd,'do_preprocess',false,'bb',false,'nonmonotone',false,'second_descent_step',false,'W_INIT',W0,'H_INIT',H0),        'Nothing!',       '#77AC30';
};

num_datasets = size(dataset_configs, 1);
num_methods  = size(method_configs,  1);
method_names = method_configs(:, 1);

r        = 3;
num_runs = 4;

% =========================================================================
% Storage for all plot data, indexed by dataset
% plot_data{d}.t_plot          — time grid for dataset d
% plot_data{d}.(method).mean   — mean accuracy curve
% plot_data{d}.dataset_name    — filename string
% plot_data{d}.plot_time_limit — per-dataset time limit
% =========================================================================
plot_data = cell(num_datasets, 1);

% =========================================================================
% MAIN LOOP OVER DATASETS
% =========================================================================
for d = 1:num_datasets
    filename       = dataset_configs{d, 1};
    gnd            = dataset_configs{d, 2};
    k              = dataset_configs{d, 3};
    plot_time_limit= dataset_configs{d, 4};
    
    % Infer r from gnd (number of unique classes)
    r = numel(unique(gnd));

    fprintf('\n========================================\n');
    fprintf('Dataset %d/%d: %s  (k=%d, r=%d)\n', d, num_datasets, filename, k, r);
    fprintf('========================================\n');

    % --- Load data ---
    X = load(filename);

    % --- Build affinity matrix ---
    init_tic = tic;
    V = X;
    [m, ~] = size(V);
    distances = pdist2(V, V, 'euclidean');
    sigma = mean(distances(:)) * 40;
    W_aff = exp(-distances.^2 / (2*sigma^2));

    for i = 1:m
        [~, idx] = sort(distances(i,:), 'ascend');
        neighbors = idx(2:k+1);
        mask = true(1, m);
        mask(neighbors) = false;
        W_aff(i, mask) = 0;
    end
    A = (W_aff + W_aff') / 2;
    init_time = toc(init_tic);

    % --- Initialize per-dataset result storage ---
    results = struct();
    for i = 1:num_methods
        mn = method_names{i};
        results.(mn).time    = cell(num_runs, 1);
        results.(mn).acc_raw = cell(num_runs, 1);
    end

    % --- Run methods ---
    for x = 1:num_runs
        fprintf('  Run %d/%d\n', x, num_runs);
        W0 = 2 * full(sqrt(mean(mean(A)) / r)) * rand(m, r);
        H0 = 2 * full(sqrt(mean(mean(A)) / r)) * rand(r, m);

        for i = 1:num_methods
            mn          = method_names{i};
            method_func = method_configs{i, 2};
            fprintf('    Method: %s\n', mn);

            [~, ~, output, acc] = method_func(A, r, gnd, W0, H0);
            output.time       = output.time       + init_time;
            output.total_time = output.total_time + init_time;

            [results.(mn).time{x}, results.(mn).acc_raw{x}] = ...
                clean_time_acc(output.total_time(:), acc(:));
        end
    end

    % --- Build common time grid ---
    max_time = 0;
    for i = 1:num_methods
        mn = method_names{i};
        for x = 1:num_runs
            max_time = max(max_time, max(results.(mn).time{x}));
        end
    end
    t_grid = linspace(0, max_time, 500);

    % --- Interpolate ---
    for i = 1:num_methods
        mn = method_names{i};
        results.(mn).acc_all = zeros(num_runs, length(t_grid));
        for x = 1:num_runs
            results.(mn).acc_all(x,:) = interpolate_with_clipping(...
                results.(mn).time{x}, results.(mn).acc_raw{x}, t_grid);
        end
    end

    % --- Trim to plot_time_limit ---
    if plot_time_limit > 0
        time_idx = t_grid <= plot_time_limit;
    else
        time_idx = true(size(t_grid));
    end
    t_plot = t_grid(time_idx);

    % --- Compute mean curves and save to plot_data ---
    plot_data{d}.dataset_name    = filename;
    plot_data{d}.plot_time_limit = plot_time_limit;
    plot_data{d}.t_plot          = t_plot;
    plot_data{d}.r               = r;
    plot_data{d}.k               = k;

    for i = 1:num_methods
        mn = method_names{i};
        plot_data{d}.(mn).mean = mean(results.(mn).acc_all(:, time_idx), 1);
        plot_data{d}.(mn).all  = results.(mn).acc_all(:, time_idx);  % per-run curves too
    end

    fprintf('  Done with %s\n', filename);
end

% =========================================================================
% PLOT ALL DATASETS (one figure per dataset)
% =========================================================================
for d = 1:num_datasets
    pd    = plot_data{d};
    t_plot = pd.t_plot;

    figure('Name', pd.dataset_name); hold on;

    for i = 1:num_methods
        mn          = method_names{i};
        plot_label  = method_configs{i, 3};
        plot_color  = method_configs{i, 4};
        plot(t_plot, pd.(mn).mean, 'Color', plot_color, 'LineWidth', 2, 'DisplayName', plot_label);
    end

    ax = gca;
    ax.FontSize = 14;
    ax.TickLabelInterpreter = 'latex';
    xlabel('$\textnormal{Time (s)}$',  'Interpreter', 'Latex', 'FontSize', 20);
    ylabel('$\textnormal{Accuracy}$',  'Interpreter', 'Latex', 'FontSize', 20);
    title(strrep(pd.dataset_name, '_', '\_'), 'FontSize', 16);
    lgd = legend('Location', 'best');
    lgd.FontSize    = 16;
    lgd.Interpreter = 'latex';
    hold off;
end

fprintf('\nAll done! Results stored in plot_data{1..%d}.\n', num_datasets);

% =========================================================================
% HELPER FUNCTIONS
% =========================================================================
function [clean_time, clean_acc] = clean_time_acc(time_vec, acc_vec)
    valid_idx = isfinite(time_vec) & isfinite(acc_vec);
    time_vec  = time_vec(valid_idx);
    acc_vec   = acc_vec(valid_idx);

    [time_vec, sort_idx] = sort(time_vec);
    acc_vec = acc_vec(sort_idx);

    [clean_time, unique_idx] = unique(time_vec, 'last');
    clean_acc = acc_vec(unique_idx);

    first_perfect = find(clean_acc >= 1.0, 1, 'first');
    if ~isempty(first_perfect)
        clean_time = clean_time(1:first_perfect);
        clean_acc  = clean_acc(1:first_perfect);
        clean_acc(end) = 1.0;
    end
end

function interp_acc = interpolate_with_clipping(time_vec, acc_vec, t_grid)
    if isempty(time_vec) || isempty(acc_vec)
        interp_acc = zeros(size(t_grid));
        return;
    end

    if time_vec(1) > 0
        time_vec = [0; time_vec(:)];
        acc_vec  = [0; acc_vec(:)];
    end

    max_time   = time_vec(end);
    interp_acc = zeros(size(t_grid));
    valid_t_idx = t_grid <= max_time;

    if any(valid_t_idx)
        interp_acc(valid_t_idx) = interp1(time_vec, acc_vec, t_grid(valid_t_idx), 'linear');
        interp_acc(~valid_t_idx) = acc_vec(end);
        interp_acc = max(0, min(interp_acc, 1.0));

        if acc_vec(end) >= 1.0
            interp_acc(t_grid >= time_vec(end)) = 1.0;
        end
    end
end