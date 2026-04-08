% clear; clc

% =========================================================================
% SYNTHETIC DATASET CONFIGURATIONS
% Each row: { n, r, plot_time_limit }
% V = W_true * W_true'  where W_true is n-by-r uniform random in [0, 5]
% Keep r << n
% =========================================================================
dataset_configs = {
    500,  5,  10;
    1000, 20, 15;
    % 1500, 300, 20;
    400, 100, 5
};

% =========================================================================
% METHOD CONFIGURATIONS
% { name, function_handle, plot_label, plot_color }
% graph_reg = -1 disables graph regularization
% No truelabel or gnd — synthetic data has no labels
% =========================================================================
method_configs = {
    'graphsnmpbb',  @(V,r,W0,H0) Graph_SNMPBB_modified(V,r,'do_preprocess',false,'W_INIT',W0,'H_INIT',H0,'graph_reg',-1),                                                                            'Everything',     '#0072BD';
    'graphsnmpbb2', @(V,r,W0,H0) Graph_SNMPBB_modified(V,r,'do_preprocess',false,'nonmonotone',false,'W_INIT',W0,'H_INIT',H0,'graph_reg',-1),                                                        'No nonmonotone', '#8516D1';
    % 'graphsnmpbb3', @(V,r,W0,H0) Graph_SNMPBB_modified(V,r,'do_preprocess',false,'bb',false,'W_INIT',W0,'H_INIT',H0,'graph_reg',-1),                                                                 'No bb',          '#2FBEEF';
    % 'graphsnmpbb4', @(V,r,W0,H0) Graph_SNMPBB_modified(V,r,'do_preprocess',false,'bb',false,'nonmonotone',false,'second_descent_step',false,'W_INIT',W0,'H_INIT',H0,'graph_reg',-1),                 'Nothing!',       '#77AC30';
};

num_datasets = size(dataset_configs, 1);
num_methods  = size(method_configs,  1);
method_names = method_configs(:, 1);

num_runs = 10;

% =========================================================================
% Storage for all plot data, indexed by dataset
% =========================================================================
plot_data = cell(num_datasets, 1);

% =========================================================================
% MAIN LOOP OVER DATASETS
% =========================================================================
for d = 1:num_datasets
    n               = dataset_configs{d, 1};
    r               = dataset_configs{d, 2};
    plot_time_limit = dataset_configs{d, 3};

    fprintf('\n========================================\n');
    fprintf('Dataset %d/%d: n=%d, r=%d\n', d, num_datasets, n, r);
    fprintf('========================================\n');

    % --- Generate synthetic data ---
    W_true = 5 * rand(n, r);
    V      = W_true * W_true';

    % --- Initialize per-dataset result storage ---
    results = struct();
    for i = 1:num_methods
        mn = method_names{i};
        results.(mn).time      = cell(num_runs, 1);
        results.(mn).resid_raw = cell(num_runs, 1);
    end

    % --- Run methods ---
    for x = 1:num_runs
        fprintf('  Run %d/%d\n', x, num_runs);

        W0 = 2 * full(sqrt(mean(mean(V)) / r)) * rand(n, r);
        H0 = 2 * full(sqrt(mean(mean(V)) / r)) * rand(r, n);

        for i = 1:num_methods
            mn          = method_names{i};
            method_func = method_configs{i, 2};
            fprintf('    Method: %s\n', mn);

            [~, ~, output, ~] = method_func(V, r, W0, H0);

            [results.(mn).time{x}, results.(mn).resid_raw{x}] = ...
                clean_time_resid(output.total_time(:), output.relres(:));
        end
    end

    % --- Build common time grid ---
    max_time = 0;
    for i = 1:num_methods
        mn = method_names{i};
        for x = 1:num_runs
            if ~isempty(results.(mn).time{x})
                max_time = max(max_time, max(results.(mn).time{x}));
            end
        end
    end
    t_grid = linspace(0, max_time, 500);

    % --- Interpolate residual ---
    for i = 1:num_methods
        mn = method_names{i};
        results.(mn).resid_all = zeros(num_runs, length(t_grid));
        for x = 1:num_runs
            results.(mn).resid_all(x,:) = interpolate_no_clip(...
                results.(mn).time{x}, results.(mn).resid_raw{x}, t_grid);
        end
    end

    % --- Trim to plot_time_limit ---
    if plot_time_limit > 0
        time_idx = t_grid <= plot_time_limit;
    else
        time_idx = true(size(t_grid));
    end
    t_plot = t_grid(time_idx);

    % --- Save to plot_data ---
    plot_data{d}.label           = sprintf('n=%d, r=%d', n, r);
    plot_data{d}.n               = n;
    plot_data{d}.r               = r;
    plot_data{d}.plot_time_limit = plot_time_limit;
    plot_data{d}.t_plot          = t_plot;

    for i = 1:num_methods
        mn = method_names{i};
        plot_data{d}.(mn).mean_resid = mean(results.(mn).resid_all(:, time_idx), 1);
        plot_data{d}.(mn).all_resid  = results.(mn).resid_all(:, time_idx);
    end

    fprintf('  Done with n=%d, r=%d\n', n, r);
end

% =========================================================================
% PLOT ALL DATASETS (one figure per dataset: residual only)
% =========================================================================
for d = 1:num_datasets
    pd     = plot_data{d};
    t_plot = pd.t_plot;

    figure('Name', pd.label); hold on;
    for i = 1:num_methods
        mn = method_names{i};
        plot(t_plot, pd.(mn).mean_resid, ...
            'Color', method_configs{i,4}, 'LineWidth', 2, 'DisplayName', method_configs{i,3});
    end
    ax = gca; ax.FontSize = 14; ax.TickLabelInterpreter = 'latex';
    xlabel('$\textnormal{Time (s)}$',       'Interpreter', 'Latex', 'FontSize', 20);
    ylabel('$\|V - WW^\top\|_F^2$',         'Interpreter', 'Latex', 'FontSize', 20);
    title(pd.label, 'FontSize', 16);
    lgd = legend('Location', 'best'); lgd.FontSize = 16; lgd.Interpreter = 'latex';
    hold off;
end

fprintf('\nAll done! Results stored in plot_data{1..%d}.\n', num_datasets);

% =========================================================================
% HELPER FUNCTIONS
% =========================================================================
function [clean_time, clean_resid] = clean_time_resid(time_vec, resid_vec)
    valid_idx = isfinite(time_vec) & isfinite(resid_vec);
    time_vec  = time_vec(valid_idx);
    resid_vec = resid_vec(valid_idx);

    [time_vec, sort_idx] = sort(time_vec);
    resid_vec = resid_vec(sort_idx);

    [clean_time, unique_idx] = unique(time_vec, 'last');
    clean_resid = resid_vec(unique_idx);
end

function interp_resid = interpolate_no_clip(time_vec, resid_vec, t_grid)
    if isempty(time_vec) || isempty(resid_vec)
        interp_resid = zeros(size(t_grid));
        return;
    end

    if time_vec(1) > 0
        time_vec  = [0; time_vec(:)];
        resid_vec = [resid_vec(1); resid_vec(:)];
    end

    max_time     = time_vec(end);
    interp_resid = zeros(size(t_grid));
    valid_t_idx  = t_grid <= max_time;

    if any(valid_t_idx)
        interp_resid(valid_t_idx)  = interp1(time_vec, resid_vec, t_grid(valid_t_idx), 'linear');
        interp_resid(~valid_t_idx) = resid_vec(end);
        interp_resid = max(0, interp_resid);
    end
end
