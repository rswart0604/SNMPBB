rng(1);
clear;
close all;
index = ssget;

vis = true;

% Select rectangular matrices with row-sizes nl <= m <= nu
nl = 600;
nu = 1400;

if vis
    bigFig = figure;
end


% Intervals for dimensions
density = index.nnz ./ (index.nrows .* index.ncols);
condit =  (index.nrows <= nu)   & ...
          (index.nrows >= nl)   & ...
          (index.ncols == index.nrows) &...
          (index.numerical_symmetry == true) &...
          (index.isReal == true) &...
          (index.xmin >= 0) &...
          (density > 0.01); % 0.2

% Linear indices
ids  = find(condit);
nids = length(ids);

% Number of problems to test
% nprob   = 38;
nprob = nids;
names   = cell(nprob,1);
iids    = zeros(nprob,1);
nnzs    = zeros(nprob,1);
ms      = zeros(nprob,1);
ns      = zeros(nprob,1);

% Store results for both algorithms
final_res_alg1 = zeros(nprob,1);
final_res_alg2 = zeros(nprob,1);
final_time_alg1 = zeros(nprob,1);
final_time_alg2 = zeros(nprob,1);
converged_alg1 = false(nprob,1);
converged_alg2 = false(nprob,1);

total_res_alg1 = {};
total_time_alg1 = {};
total_res_alg2 = {};
total_time_alg2 = {};

fprintf('***************** Algorithm Comparison ********************* \n');
fprintf('*         Test matrices                                      \n');
fprintf('*         Number of Problems: %i                             \n',nprob);
% fprintf('*         Sizes: n = m, %i <= n <= %i                       \n',nl,nu);
fprintf('*********************************************************** \n\n');

% Main loop
for i = 1:5
    fprintf('Problem %i/%i: ', i, nprob);
    
    % Loading problem and dimensions
    tic;
    Prob = ssget(ids(i));
    A = Prob.A;
    if ~issymmetric(A)
        continue;
    end
    [m,n] = size(A);
    
    do_rank_reduce = true;
    if do_rank_reduce
        col_norms = vecnorm(A,2,1);
        [~, idx] = sort(col_norms,'descend');
        rank_choice = 300;%round(numel(col_norms)/2);
        % rank_choice = 300;

        X = max(A(:,idx(1:rank_choice)),0);
        A = X*X';
    end

    spd = 1 ./ sqrt(max(abs(A),[],2));
    D = spdiags(d,0,n,n);
    A = D*A*D;

    
        
    % Store problem information
    iids(i) = Prob.id;
    names{i} = Prob.name;
    nnzs(i) = nnz(A);
    ms(i) = m;
    ns(i) = n;
    
    fprintf('%s (size: %i, nnz: %i, density: %.5f)\n', names{i}, m, ...
        nnzs(i), nnzs(i)/m^2);
    
    % Run Algorithm 1: LAI_SNMPBB
    
    A = A/normest(A);
    sym_weight = max(A(:));
    toc;
    k = round(rank_choice * 2/3);
    H = 2 * full(sqrt(mean(mean(A)) / k)) * rand(n, k);
    tic;
    [~, ~, output1, ~] = Graph_SNMPBB_modified(A, k, 'H_INIT', H', 'W_INIT', H, 'verbose', 1, 'tol', 1e-8, 'graph_reg', -1);%, 'INNER_MAX_ITER', 10);
    toc;
    % [~, ~, output1] = LAI_SNMPBB(A, k, 'sym_weight', sym_weight, 'H_INIT', H', 'W_INIT', H, ...
                                  % 'p', 60, 'do_AQB', true, 'verbose', 1, 'tol', 1e-8, 'INNER_MAX_ITER', 3);
    % return;
    
    % Run Algorithm 2: LAI_SymPGNCG
    tic;
    [~, ~, output2, ~] = Graph_SNMPBB_modified(A, k, 'H_INIT', H', 'W_INIT', H, 'verbose', 1, 'tol', 1e-8, 'nonmonotone', false, 'graph_reg', -1);%, 'INNER_MAX_ITER', 10);
    toc;% init.H = H;
    % [~, output2, ~] = LAI_SymPGNCG(A, k, 'p', 60, 'do_AQB', true, 'tol', 1e-8, 'init',init);
    
    % Extract final residuals
    fo1 = output1.relres(output1.relres ~= 0);
    fo2 = output2.relres(output2.relres ~= 0);

    total_res_alg1{i} = fo1;
    total_time_alg1{i} = output1.total_time(1:length(fo1));
    total_res_alg2{i} = fo2;
    total_time_alg2{i} = output2.total_time(1:length(fo2));
    % bar = cumsum(output2.time);
    % total_time_alg2{i} = bar(1:length(fo2));


    plotsPerFig = 16;
    rows = 4; cols = 4;
    
    figIdx = ceil(i / plotsPerFig);
    figure(figIdx);
    subplotIdx = mod(i-1, plotsPerFig) + 1;
    
    if subplotIdx == 1
        figure('Name', sprintf('Convergence %d–%d', ...
            (figIdx-1)*plotsPerFig+1, min(figIdx*plotsPerFig,nprob)));
    end
    
    subplot(rows, cols, subplotIdx);


    % subplot(ceil(sqrt(nprob)), ceil(sqrt(nprob)), i);
    plot(total_time_alg1{i}, total_res_alg1{i}); hold on;
    plot(total_time_alg2{i}, total_res_alg2{i}); hold on;
    if (i==1)
        legend('everything', 'no nonmonotone');
    end
    title(sprintf('Problem %d', i));
    set(gca,'YScale','log');   % optional, if residual is small

        % clf;
        % plot(total_time_alg1{i}, total_res_alg1{i}); hold on;
        % plot(total_time_alg2{i}, total_res_alg2{i});
        % disp("foo");
    % end


    final_res_alg1(i) = fo1(end);
    final_res_alg2(i) = fo2(end);
    
    final_time_alg1(i) = output1.total_time(length(fo1));
    final_time_alg2(i) = sum(output2.time);
    
    converged_alg1(i) = fo1(end) < 1e-8;
    converged_alg2(i) = fo2(end) < 1e-8;
end

%% ===== VISUALIZATION =====

if vis
    % saveas(bigFig, 'suitesparse_test/all_plots.png');
    % or higher resolution:
    % exportgraphics(bigFig, 'suitesparse_test/all_plots.pdf', 'Resolution', 300);
end



% 1. Performance Profile
figure('Position', [100, 100, 1200, 400]);

subplot(1,3,1)
% Calculate performance ratios for residuals
best_res = min([final_res_alg1, final_res_alg2], [], 2);
ratio_alg1 = final_res_alg1 ./ best_res;
ratio_alg2 = final_res_alg2 ./ best_res;

% Performance profile
tau = logspace(0, 2, 100);  % Performance ratios from 1 to 100
perf_alg1 = zeros(size(tau));
perf_alg2 = zeros(size(tau));

for k = 1:length(tau)
    perf_alg1(k) = sum(ratio_alg1 <= tau(k)) / nprob;
    perf_alg2(k) = sum(ratio_alg2 <= tau(k)) / nprob;
end

semilogx(tau, perf_alg1, 'b-', 'LineWidth', 2); hold on;
semilogx(tau, perf_alg2, 'r--', 'LineWidth', 2);
xlabel('Performance Ratio \tau', 'FontSize', 12);
ylabel('Fraction of Problems Solved', 'FontSize', 12);
title('Performance Profile (Residual)', 'FontSize', 14);
legend('LAI\_SNMPBB', 'LAI\_SymPGNCG', 'Location', 'southeast');
grid on;

% 2. Scatter plot
% subplot(1,3,2)
% loglog(final_res_alg1, final_res_alg2, 'bo', 'MarkerSize', 8, 'LineWidth', 1.5);
% hold on;
% min_val = min([final_res_alg1; final_res_alg2]);
% max_val = max([final_res_alg1; final_res_alg2]);
% loglog([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 1.5);
% xlabel('LAI\_SNMPBB Final Residual', 'FontSize', 12);
% ylabel('LAI\_SymPGNCG Final Residual', 'FontSize', 12);
% title('Final Residual Comparison', 'FontSize', 14);
% grid on;
% axis equal tight;
% 
% 3. Bar chart of wins
% subplot(1,3,3)
% wins_alg1 = sum(final_res_alg1 < final_res_alg2);
% wins_alg2 = sum(final_res_alg2 < final_res_alg1);
% ties = nprob - wins_alg1 - wins_alg2;
% 
% bar([wins_alg1, wins_alg2, ties]);
% set(gca, 'XTickLabel', {'SNMPBB Wins', 'SymPGNCG Wins', 'Ties'});
% ylabel('Number of Problems', 'FontSize', 12);
% title('Algorithm Wins', 'FontSize', 14);
% grid on;

% % % % %% ===== SUMMARY TABLE =====
% % % % fprintf('\n========== SUMMARY STATISTICS ==========\n');
% % % % fprintf('Algorithm         | Mean Res  | Median Res | Min Res   | Max Res   | Wins | Converged\n');
% % % % fprintf('------------------|-----------|------------|-----------|-----------|------|-----------\n');
% % % % fprintf('LAI_SNMPBB        | %.2e | %.2e   | %.2e  | %.2e  | %3i  | %3i/%i\n', ...
% % % %     mean(final_res_alg1), median(final_res_alg1), min(final_res_alg1), ...
% % % %     max(final_res_alg1), wins_alg1, sum(converged_alg1), nprob);
% % % % fprintf('LAI_SymPGNCG      | %.2e | %.2e   | %.2e  | %.2e  | %3i  | %3i/%i\n', ...
% % % %     mean(final_res_alg2), median(final_res_alg2), min(final_res_alg2), ...
% % % %     max(final_res_alg2), wins_alg2, sum(converged_alg2), nprob);
% % % % fprintf('=========================================\n\n');
% % % % 
% % % % % Detailed results table
% % % % fprintf('\n========== DETAILED RESULTS ==========\n');
% % % % fprintf('%-4s | %-30s | %12s | %12s | %10s\n', 'No.', 'Matrix Name', 'SNMPBB', 'SymPGNCG', 'Winner');
% % % % fprintf('-----|--------------------------------|--------------|--------------|----------\n');
% % % % for i = 1:nprob
% % % %     if final_res_alg1(i) < final_res_alg2(i)
% % % %         winner = 'SNMPBB';
% % % %     elseif final_res_alg2(i) < final_res_alg1(i)
% % % %         winner = 'SymPGNCG';
% % % %     else
% % % %         winner = 'Tie';
% % % %     end
% % % %     fprintf('%4i | %-30s | %.4e | %.4e | %s\n', ...
% % % %         i, names{i}, final_res_alg1(i), final_res_alg2(i), winner);
% % % % end
% % % % fprintf('======================================\n');
% % % % 
% % % % % % Save results
% % % % % save('algorithm_comparison_results.mat', 'names', 'final_res_alg1', 'final_res_alg2', ...
% % % % %      'final_time_alg1', 'final_time_alg2', 'converged_alg1', 'converged_alg2', ...
% % % % %      'ms', 'ns', 'nnzs');
% % % % save('new_algorithm_comparison_results.mat', 'names', 'total_res_alg1', 'total_res_alg2',...
% % % %     'total_time_alg1', 'total_time_alg2','ms', 'ns', 'nnzs');



%%
nprob = numel(names);

% mat_size = zeros(nprob, 1);
% mat_nnz  = zeros(nprob, 1);
% 
% for i = 1:nprob
%     A = A_cell{i};
%     mat_size(i) = size(A, 1);
%     mat_nnz(i)  = nnz(A);
% end

% Create table
T = table((1:nprob)', names(:), ns, nnzs, ...
    final_res_alg1(:), final_res_alg2(:), ...
    'VariableNames', {'No', 'Matrix', 'Size', 'NNZ', 'SNMPBB', 'SymPGNCG'});

% Write to CSV
% writetable(T, 'comparison_results2.csv');



% if you want to just get this, just open "all_var.mat" and then run the
% below section

% performance profile stuff
tol_eff = 1e-6;  % "effectively converged" threshold — tune this to match your plots
for i = 1:34
if isempty(total_res_alg1{i}), continue; end
% --- Alg 1 ---
idx1 = find(total_res_alg1{i} <= tol_eff, 1, 'first');
if ~isempty(idx1)
final_time_alg1(i) = total_time_alg1{i}(idx1);
final_res_alg1(i)  = total_res_alg1{i}(idx1);
end
% --- Alg 2 ---
idx2 = find(total_res_alg2{i} <= tol_eff, 1, 'first');
if ~isempty(idx2)
final_time_alg2(i) = total_time_alg2{i}(idx2);
final_res_alg2(i)  = total_res_alg2{i}(idx2);
end
end
ran = ~cellfun('isempty', names(1:34));
% Time-to-convergence profile
T_time = [final_time_alg1(ran), final_time_alg2(ran)];
ex     = ones(sum(ran), 2);  % all runs treated as valid
% Or mark as failed if never hit tol_eff:
% ex(:,1) = final_res_alg1(ran) <= tol_eff;
% ex(:,2) = final_res_alg2(ran) <= tol_eff;
perf_ext_fnc(ex, T_time, {'LAI\_SNMPBB', 'LAI\_SymPGNCG'});
title('Performance Profile — Time to Convergence');

% Residual performance profile
T_res = [final_res_alg1(ran), final_res_alg2(ran)];
ex    = ones(sum(ran), 2);

perf_ext_fnc(ex, T_res, {'LAI\_SNMPBB', 'LAI\_SymPGNCG'});
title('Performance Profile — Final Residual');