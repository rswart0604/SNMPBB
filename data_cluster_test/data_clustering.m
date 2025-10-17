% load("ORL.mat");
% r = 40;
% X = data;
% gnd = label;
% k = 3;
% plot_name = "orl_mean";

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

load("2k2k.mat")
X = fea;
r = 10;
k = 42;
% plot_name = "2k2k_mean";

num_runs = 10;

% storage for accuracy curves
time_symcluster1 = cell(num_runs,1);
time_symcluster2 = cell(num_runs,1);
time_symcluster3 = cell(num_runs,1);
time_symcluster4 = cell(num_runs,1);

acc_symcluster1_raw = cell(num_runs,1);
acc_symcluster2_raw = cell(num_runs,1);
acc_symcluster3_raw = cell(num_runs,1);
acc_symcluster4_raw = cell(num_runs,1);

[m,~] = size(X);

for x = 1:num_runs
    W0 = 2 * full(sqrt(mean(mean(X)) / r)) * rand(m, r);
    H0 = 2 * full(sqrt(mean(mean(X)) / r)) * rand(r,m);    
    params.truelabel = gnd;
    params.Hinit = W0;
    params.Winit = H0;
            
    params.alg = 'graph_snmpbb';
    [H_symcluster1,output_symcluster1,acc_symcluster1] = symnmf_cluster(X,r,params);
    params.alg = 'anls';
    [H_symcluster2,output_symcluster2,acc_symcluster2] = symnmf_cluster(X,r,params);
    params.alg = 'pgd';
    [H_symcluster3,output_symcluster3,acc_symcluster3] = symnmf_cluster(X,r,params);
    % params.alg = 'newton';
    % [H_symcluster4,output_symcluster4,acc_symcluster4] = symnmf_cluster(X,r,params);
    
    [time_symcluster1{x}, acc_symcluster1_raw{x}] = clean_time_acc(output_symcluster1.total_time(:), acc_symcluster1(:));
    [time_symcluster2{x}, acc_symcluster2_raw{x}] = clean_time_acc(output_symcluster2.total_time(:), acc_symcluster2(:));
    [time_symcluster3{x}, acc_symcluster3_raw{x}] = clean_time_acc(output_symcluster3.total_time(:), acc_symcluster3(:));
    % [time_symcluster4{x}, acc_symcluster4_raw{x}] = clean_time_acc(output_symcluster4.total_time(:), acc_symcluster4(:));
end

% Find the maximum time across all methods and runs
max_time = 0;
for x = 1:num_runs
    max_time = max(max_time, max([time_symcluster1{x}; time_symcluster2{x}; time_symcluster3{x}]));
end

% Use a common time grid
t_grid = linspace(0, max_time, 500);

% Storage for interpolated curves
acc_symcluster1_all = zeros(num_runs, length(t_grid));
acc_symcluster2_all = zeros(num_runs, length(t_grid));
acc_symcluster3_all = zeros(num_runs, length(t_grid));

% Interpolation for each run with proper clipping
for x = 1:num_runs
    % Graph SNMPBB
    
    % ANLS
    % acc_anls_all(x,:) = interpolate_with_clipping(time_anls{x}, acc_anls_raw{x}, t_grid);
    
    % Newton
    % acc_newton_all(x,:) = interpolate_with_clipping(time_newton{x}, acc_newton_raw{x}, t_grid);
        
    % PGD
    % acc_pgd_all(x,:) = interpolate_with_clipping(time_pgd{x}, acc_pgd_raw{x}, t_grid);
    
    % Symcluster variants
    acc_symcluster1_all(x,:) = interpolate_with_clipping(time_symcluster1{x}, acc_symcluster1_raw{x}, t_grid);
    acc_symcluster2_all(x,:) = interpolate_with_clipping(time_symcluster2{x}, acc_symcluster2_raw{x}, t_grid);
    acc_symcluster3_all(x,:) = interpolate_with_clipping(time_symcluster3{x}, acc_symcluster3_raw{x}, t_grid);
end

% Add this parameter to control plot time range (in seconds)
plot_time_limit = 3; % Show only first 30 seconds (adjust as needed)

% Find the indices corresponding to the time limit
if plot_time_limit > 0
    time_idx = t_grid <= plot_time_limit;
    t_plot = t_grid(time_idx);
else
    % If plot_time_limit is 0 or negative, show all data
    time_idx = true(size(t_grid));
    t_plot = t_grid;
end

figure; hold on
snmpbb_data = mean(acc_symcluster1_all(:,time_idx),1);
anls_data = mean(acc_symcluster2_all(:,time_idx),1);
pgd_data = mean(acc_symcluster3_all(:,time_idx),1);
plot(t_plot, snmpbb_data,'LineWidth',2)
plot(t_plot, anls_data,'LineWidth',2)
plot(t_plot, pgd_data,'LineWidth',2)
ax = gca;
ax.FontSize = 14;
ax.TickLabelInterpreter = 'latex';
xlabel('$\textnormal{Time (s)}$', 'Interpreter', 'Latex', 'Fontsize', 20);
ylabel('$\textnormal{Accuracy}$', 'Interpreter', 'Latex', 'Fontsize', 20);
lgd = legend("Graph SNMPBB","ANLS",...
    "PGD",'Location','best');
lgd.FontSize = 16;
lgd.Interpreter = 'latex';
% Optional: Set x-axis limits explicitly
if plot_time_limit > 0
    xlim([0, plot_time_limit]);
end
export_matrix = [t_plot; snmpbb_data; anls_data; pgd_data]';
writematrix(export_matrix, 'data/2k2k.csv');


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