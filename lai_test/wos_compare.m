load("T_large.mat")
Y = readlines("YL1_large.txt");
trueLabels = str2double(Y);


our_ari = 0;
their_ari = 0;


for i=1:10
    [W , H, output1] = LAI_SNMPBB(T, 7,'sym_weight',0.01,'p',40,'do_AQB',true,'verbose',1,'tol',1e-5);
    [H4,output4,U4] = LAI_SymPGNCG(T,7,'p',40,'do_AQB',true);
    [~, predLabels] = max(H4, [], 2);
    their_ari(i) = adjustedRandIndex(trueLabels, predLabels(1:end));

    our_outputs{i} = output1;
    their_outputs{i} = output4;

    numerator_squared = norm(T,'fro')^2 ...
    - 2*trace(W' * T * W) ...
    + trace((W' * W)^2);
    relative_error = sqrt(numerator_squared)/norm(T,'fro');
    our_err(i) = relative_error;

    numerator_squared = norm(T,'fro')^2 ...
    - 2*trace(H4' * T * H4) ...
    + trace((H4' * H4)^2);
    relative_error = sqrt(numerator_squared)/norm(T,'fro');
    their_err(i) = relative_error;


    [~, predLabels] = max(W, [], 2);   % n×1 predicted cluster labels
    our_ari(i) = adjustedRandIndex(trueLabels, predLabels(1:end));
    i
end

% Set plotting parameters
plot_start_time = 9.6;
plot_end_time = [];   % Leave empty to use max time, or set a specific end time

% Find the maximum time across all runs for both methods
max_time = 0;
for i = 1:10
    time_our = cumsum(our_outputs{i}.time(our_outputs{i}.time ~= 0));
    time_their = cumsum(their_outputs{i}.time);
    max_time = max([max_time, max(time_our), max(time_their)]);
end

% Set end time
if isempty(plot_end_time)
    plot_end_time = max_time;
end

% Use a common time grid
t_grid = linspace(plot_start_time, plot_end_time, 500);

% Storage for interpolated curves
relres_our_all = zeros(10, length(t_grid));
relres_their_all = zeros(10, length(t_grid));

% Interpolation for each run
for i = 1:10
    % LAI-SNMPBB (our method)
    time_our = cumsum(our_outputs{i}.time(our_outputs{i}.time ~= 0));
    relres_our = our_outputs{i}.relres(1:numel(time_our));
    relres_our_all(i,:) = interpolate_with_bounds(time_our, relres_our, t_grid, plot_start_time);
    
    % LAI-SymPGNCG (their method) 
    time_their = cumsum(their_outputs{i}.time);
    relres_their = their_outputs{i}.relres;
    relres_their_all(i,:) = interpolate_with_bounds(time_their, relres_their, t_grid, plot_start_time);
end

% Plot averaged results (only the time range of interest)
figure; hold on
time_idx = (t_grid >= plot_start_time) & (t_grid <= plot_end_time);
h1 = plot(t_grid(time_idx), mean(relres_our_all(:, time_idx), 1), 'r', 'LineWidth', 2);
h2 = plot(t_grid(time_idx), mean(relres_their_all(:, time_idx), 1), 'b', 'LineWidth', 2);
export_matrix = [t_grid(time_idx); mean(relres_our_all(:, time_idx), 1); mean(relres_their_all(:, time_idx))]';
writematrix(export_matrix, 'data/wos_compare.csv');


% Optional: Add confidence bands (standard deviation)
% our_std = std(relres_our_all(:, time_idx), 0, 1);
% their_std = std(relres_their_all(:, time_idx), 0, 1);
% t_plot = t_grid(time_idx);
% fill([t_plot, fliplr(t_plot)], [mean(relres_our_all(:,time_idx),1) - our_std, fliplr(mean(relres_our_all(:,time_idx),1) + our_std)], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
% fill([t_plot, fliplr(t_plot)], [mean(relres_their_all(:,time_idx),1) - their_std, fliplr(mean(relres_their_all(:,time_idx),1) + their_std)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

ax = gca;
ax.FontSize = 14;
ax.TickLabelInterpreter = 'latex';
xlabel('$\textnormal{Time (s)}$', 'Interpreter', 'latex', 'FontSize', 20);
ylabel('$\textnormal{Relative residual}$', 'Interpreter', 'latex', 'FontSize', 20);
legend([h1 h2], {'LAI-SNMPBB','LAI-SymPGNCG'}, 'Location', 'best', 'FontSize', 16, 'Interpreter', 'latex');
% print('plots/wos_relres_averaged', '-dpng')

% Helper function for interpolation with bounds
function interp_vals = interpolate_with_bounds(time_vec, vals_vec, t_grid, plot_start_time)
    if isempty(time_vec) || isempty(vals_vec)
        interp_vals = ones(size(t_grid));  % Default high value for relative residual
        return;
    end
    
    % Ensure time starts at the specified start time or 0
    if time_vec(1) > plot_start_time
        time_vec = [max(0, plot_start_time); time_vec(:)];
        vals_vec = [vals_vec(1); vals_vec(:)];  % Use first value for start time
    end
    
    % Find maximum time we have data for
    max_time = time_vec(end);
    
    % Initialize output
    interp_vals = ones(size(t_grid)) * vals_vec(end);  % Fill with last known value
    
    % Only interpolate up to the maximum time we have data for
    valid_t_idx = t_grid <= max_time;
    
    if any(valid_t_idx)
        % Interpolate the valid portion
        interp_vals(valid_t_idx) = interp1(time_vec, vals_vec, t_grid(valid_t_idx), 'linear');
        
        % For times beyond our data, use the last known value (already set above)
        
        % Ensure positive values (relative residual should be >= 0)
        interp_vals = max(0, interp_vals);
    end
end