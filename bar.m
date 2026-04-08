T_budgets = [1];   % seconds
n_prob    = 34;

%% --- 1. Find valid problems (both algorithms have data) ---
ran = false(n_prob,1);

for i = 1:n_prob
    ran(i) = ...
        ~isempty(total_time_alg1{i}) && ~isempty(total_res_alg1{i}) && ...
        ~isempty(total_time_alg2{i}) && ~isempty(total_res_alg2{i});
end

ran_idx = find(ran);
n_ran   = numel(ran_idx);

if n_ran == 0
    error('No problems have data for both algorithms.');
end

%% --- 2. Create figure ---
% figure('Position',[100,100,420*numel(T_budgets),400]);

for b = 1:numel(T_budgets)

    T_budget = T_budgets(b);

    % Preallocate
    res_alg1 = inf(n_ran,1);
    res_alg2 = inf(n_ran,1);
    ex       = false(n_ran,2);   % success flags

    %% --- 3. Extract residuals at budget ---
    for j = 1:n_ran

        i = ran_idx(j);

        % ---- Algorithm 1 ----
        t1 = total_time_alg1{i};
        r1 = total_res_alg1{i};

        idx1 = find(t1 <= T_budget, 1, 'last');

        if ~isempty(idx1) && isfinite(r1(idx1))
            res_alg1(j) = r1(idx1);
            ex(j,1)     = true;
        end

        % ---- Algorithm 2 ----
        t2 = total_time_alg2{i};
        r2 = total_res_alg2{i};

        idx2 = find(t2 <= T_budget, 1, 'last');

        if ~isempty(idx2) && isfinite(r2(idx2))
            res_alg2(j) = r2(idx2);
            ex(j,2)     = true;
        end
    end

    %% --- 4. Sanity check (important for debugging broken plots) ---
    if all(~ex(:,1)) || all(~ex(:,2))
        warning('Budget %d: One algorithm has zero successful runs.', T_budget);
    end

    %% --- 5. Build matrix for performance profile ---
    T_res = [res_alg1, res_alg2];

    %% --- 6. Plot ---
    subplot(1, numel(T_budgets), b);

    try
        perf_ext_fnc(ex, T_res, {'LAI\_SNMPBB','LAI\_SymPGNCG'});
    catch ME
        warning('perf_ext_fnc failed at budget %d: %s', T_budget, ME.message);
        continue;
    end

    title(sprintf('Budget = %ds', T_budget));

    % Use safe TeX (not LaTeX unless you enforce it everywhere)
    xlabel('Performance Ratio \tau');
    ylabel('Fraction of Problems');

    grid on;
end

sgtitle('Time Budget Performance Profiles');