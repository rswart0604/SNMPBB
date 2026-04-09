% clear; clc

% =========================================================================
% DATASET CONFIGURATIONS
% Each row: { 'filename', r, plot_time_limit, preprocess_fn }
%
% preprocess_fn is a function handle that takes the raw loaded struct and
% returns [X, gnd]. Use @default_load for datasets with no special steps.
%
% Variables available inside each preprocess_fn via the loaded struct:
%   s.fea, s.gnd, s.data, s.label, s.Y, etc. (whatever the .mat contains)
% =========================================================================
dataset_configs = {
    'ORL.mat',         40,  2.5,   @(s) deal(s.data, s.label);  % 400
    % 'ORL.mat', 40, 2.5, @(s) preprocess_orl(s)
    'COIL20.mat',      20,  2.5,   @(s) deal(s.X,  s.Y);  % 1440
    'Isolet1.mat',     26,  10,   @(s) deal(s.fea,  s.gnd);  % 1560
    '2k2k.mat',        10,  10,   @(s) deal(s.fea,  s.gnd);  % 4000
    'Reuters21578.mat',20,  10,   @(s) top20_preprocess(s.fea, s.gnd);  % 8293
    % 'TDT2_all.mat',    20,  10,   @(s) top20_preprocess(s.fea, s.gnd);
};

% =========================================================================
% METHOD CONFIGURATIONS
% { name, alg_param, plot_label, plot_color, extra_params_struct }
% =========================================================================
method_configs = {
    'graphsnmpbb_modified',  'modified_graph_snmpbb', 'Plain',          '#0072BD', struct('bb',true,  'second_descent_step',true,  'nonmonotone',true);
    'graphsnmpbb_modified1', 'modified_graph_snmpbb', 'No nonmonotone', '#D95319', struct('bb',true,  'second_descent_step',true,  'nonmonotone',false);
    'graphsnmpbb_modified2', 'modified_graph_snmpbb', 'No bb',          '#77AC30', struct('bb',false, 'second_descent_step',true,  'nonmonotone',true);
    'graphsnmpbb_modified3', 'modified_graph_snmpbb', 'Nothing',        '#8516D1', struct('bb',false, 'second_descent_step',false, 'nonmonotone',false);
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
    filename        = dataset_configs{d, 1};
    r               = dataset_configs{d, 2};
    plot_time_limit = dataset_configs{d, 3};
    preprocess_fn   = dataset_configs{d, 4};

    fprintf('\n========================================\n');
    fprintf('Dataset %d/%d: %s  (r=%d)\n', d, num_datasets, filename, r);
    fprintf('========================================\n');

    % --- Load and preprocess ---
    s = load(filename);
    [X, gnd] = preprocess_fn(s);
    [m, ~] = size(X);
    

    % --- Initialize per-dataset result storage ---
    results = struct();
    for i = 1:num_methods
        mn = method_names{i};
        results.(mn).time        = cell(num_runs, 1);
        results.(mn).acc_raw     = cell(num_runs, 1);
        results.(mn).relres_time = cell(num_runs, 1);
        results.(mn).relres_raw  = cell(num_runs, 1);
    end

    % --- Run methods ---
    for x = 1:num_runs
        fprintf('  Run %d/%d\n', x, num_runs);
        tic;

        W0 = 2 * full(sqrt(mean(mean(X)) / r)) * rand(m, r);
        H0 = 2 * full(sqrt(mean(mean(X)) / r)) * rand(r, m);

        for i = 1:num_methods
            mn         = method_names{i};
            alg_param  = method_configs{i, 2};
            extra      = method_configs{i, 5};
            fprintf('    Method: %s\n', mn);

            % Build params struct
            params           = struct();
            params.truelabel = gnd;
            params.Hinit     = W0;
            params.Winit     = H0;
            params.alg       = alg_param;
            params.kk = 2*floor(log2(m)/log(2)) + 1;
            params.s = 1;

            % Merge extra params if alg is modified
            if strcmp(alg_param, 'modified_graph_snmpbb')
                fields = fieldnames(extra);
                for j = 1:numel(fields)
                    params.(fields{j}) = extra.(fields{j});
                end
            end

            [~, output, acc] = symnmf_cluster(X, r, params);

            [results.(mn).time{x}, results.(mn).acc_raw{x}] = ...
                clean_time_acc(output.total_time(:), acc(:));
            [results.(mn).relres_time{x}, results.(mn).relres_raw{x}] = ...
                clean_time_acc(output.total_time(:), output.relres(:));
        end

        toc;
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

    % --- Interpolate accuracy and residual ---
    for i = 1:num_methods
        mn = method_names{i};
        results.(mn).acc_all    = zeros(num_runs, length(t_grid));
        results.(mn).relres_all = zeros(num_runs, length(t_grid));
        for x = 1:num_runs
            results.(mn).acc_all(x,:) = interpolate_with_clipping(...
                results.(mn).time{x}, results.(mn).acc_raw{x}, t_grid);
            results.(mn).relres_all(x,:) = interpolate_with_clipping(...
                results.(mn).relres_time{x}, results.(mn).relres_raw{x}, t_grid);
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
    plot_data{d}.dataset_name    = filename;
    plot_data{d}.plot_time_limit = plot_time_limit;
    plot_data{d}.t_plot          = t_plot;
    plot_data{d}.r               = r;

    for i = 1:num_methods
        mn = method_names{i};
        plot_data{d}.(mn).mean_acc    = mean(results.(mn).acc_all(:, time_idx), 1);
        plot_data{d}.(mn).mean_relres = mean(results.(mn).relres_all(:, time_idx), 1);
        plot_data{d}.(mn).all_acc     = results.(mn).acc_all(:, time_idx);
        plot_data{d}.(mn).all_relres  = results.(mn).relres_all(:, time_idx);
    end

    fprintf('  Done with %s\n', filename);
end

% =========================================================================
% PLOT ALL DATASETS (two figures per dataset: accuracy + residual)
% =========================================================================
for d = 1:num_datasets
    pd     = plot_data{d};
    t_plot = pd.t_plot;
    name   = strrep(pd.dataset_name, '_', '\_');

    % --- Accuracy ---
    figure('Name', [pd.dataset_name ' - Accuracy']); hold on;
    for i = 1:num_methods
        mn = method_names{i};
        plot(t_plot, pd.(mn).mean_acc, ...
            'Color', method_configs{i,4}, 'LineWidth', 2, 'DisplayName', method_configs{i,3});
    end
    ax = gca; ax.FontSize = 14; ax.TickLabelInterpreter = 'latex';
    xlabel('$\textnormal{Time (s)}$',  'Interpreter', 'Latex', 'FontSize', 20);
    ylabel('$\textnormal{Accuracy}$',  'Interpreter', 'Latex', 'FontSize', 20);
    title(name, 'FontSize', 16);
    lgd = legend('Location', 'best'); lgd.FontSize = 16; lgd.Interpreter = 'latex';
    hold off;

    % --- Relative Residual ---
    % figure('Name', [pd.dataset_name ' - Residual']); hold on;
    % for i = 1:num_methods
    %     mn = method_names{i};
    %     plot(t_plot, pd.(mn).mean_relres, ...
    %         'Color', method_configs{i,4}, 'LineWidth', 2, 'DisplayName', method_configs{i,3});
    % end
    % ax = gca; ax.FontSize = 14; ax.TickLabelInterpreter = 'latex';
    % xlabel('$\textnormal{Time (s)}$',           'Interpreter', 'Latex', 'FontSize', 20);
    % ylabel('$\textnormal{Relative Residual}$',  'Interpreter', 'Latex', 'FontSize', 20);
    % title(name, 'FontSize', 16);
    % lgd = legend('Location', 'best'); lgd.FontSize = 16; lgd.Interpreter = 'latex';
    % hold off;
end

fprintf('\nAll done! Results stored in plot_data{1..%d}.\n', num_datasets);

% =========================================================================
% PREPROCESSING HELPERS
% =========================================================================

% Keep top-20 largest clusters (used for Reuters / TDT2)
function [X_out, gnd_out] = top20_preprocess(X_in, gnd_in)
    [clusterLabels, ~, idx] = unique(gnd_in);
    clusterSizes = accumarray(idx, 1);
    [~, sortIdx] = sort(clusterSizes, 'descend');
    top20_labels = clusterLabels(sortIdx(1:20));
    mask = ismember(gnd_in, top20_labels);
    X_out   = X_in(mask, :);
    gnd_out = gnd_in(mask);
end

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

    max_time    = time_vec(end);
    interp_acc  = zeros(size(t_grid));
    valid_t_idx = t_grid <= max_time;

    if any(valid_t_idx)
        interp_acc(valid_t_idx)  = interp1(time_vec, acc_vec, t_grid(valid_t_idx), 'linear');
        interp_acc(~valid_t_idx) = acc_vec(end);
        interp_acc = max(0, min(interp_acc, 1.0));

        if acc_vec(end) >= 1.0
            interp_acc(t_grid >= time_vec(end)) = 1.0;
        end
    end
end

function [X, gnd] = preprocess_orl(s)
    label = s.label; data = s.data;
    [~, sortIdx] = sort(label);
    dataSorted = data(sortIdx, :);
    
    % Now every block of 10 rows is one class (all 1s, then all 2s...)
    mask = repmat([true(5,1); false(5,1)], 40, 1);
    X = dataSorted(mask, :);
    gnd = label(sortIdx(mask));
    size(X)
end