clear; clc;

m_vals = [5000];
r_vals = [5000,120];
num_runs = 3;


for i_=1:numel(m_vals)
    for j_=1:numel(r_vals)
        m = m_vals(i_);
        r = r_vals(j_);
        outputs_list_ours{i_,j_} = cell(1,num_runs);
        outputs_list_theirs{i_,j_} = cell(1,num_runs);
        for i = 1:num_runs
            W_old = max(0, randn(m, r));
            V = W_old * W_old';
            sym_weight = max(V(:))*1e-3;
            [~, ~, output1] = LAI_SNMPBB(V, 100,'sym_weight',sym_weight,'p',2,'do_AQB',true,'verbose',1,'tol',1e-8);
            [~,output2,~] = LAI_SymPGNCG(V,100,'p',2,'do_AQB',true,'tol',1e-8);
            outputs_list_ours{i_,j_}{i} = output1;
            outputs_list_theirs{i_,j_}{i} = output2;


            fo1 = output1.relres(output1.relres ~= 0);
            fo2 = output2.relres(output2.relres ~= 0);
        
            total_res_alg1{i} = fo1;
            total_time_alg1{i} = output1.total_time(1:length(fo1));
            total_res_alg2{i} = fo2;
            bar = cumsum(output2.time);
            total_time_alg2{i} = bar(1:length(fo2));
        
            if true
                clf;
                plot(total_time_alg1{i}, total_res_alg1{i}); hold on;
                plot(total_time_alg2{i}, total_res_alg2{i});
                disp("foo");
            end

        end

    end
end

results = [];

for i_ = 1:numel(m_vals)
    for j_ = 1:numel(r_vals)
        m = m_vals(i_);
        r = r_vals(j_);

        ours_times = zeros(1,num_runs);
        ours_rels  = zeros(1,num_runs);
        their_times = zeros(1,num_runs);
        their_rels  = zeros(1,num_runs);

        for run = 1:num_runs
            out1 = outputs_list_ours{i_, j_}{run};
            out2 = outputs_list_theirs{i_, j_}{run};

            ours_times(run)  = max(out1.total_time);
            ours_rels(run)   = out1.relres(find(out1.relres ~= 0, 1, 'last'));

            their_times(run) = max(cumsum(out2.time));
            their_rels(run)  = out2.relres(find(out2.relres ~= 0, 1, 'last'));
        end

        results(end+1).m = m;
        results(end).r = r;

        results(end).ours_time_mean  = mean(ours_times);
        results(end).ours_rel_mean   = mean(ours_rels);

        results(end).their_time_mean = mean(their_times);
        results(end).their_rel_mean  = mean(their_rels);
    end
end

T = struct2table(results);
disp(T)

for i = 1:height(T)
    fprintf('%d & %d & \\num{%.4f} & \\num{%.2e} & \\num{%.4f} & \\num{%.2e} \\\\\n', ...
        T.m(i), T.r(i), ...
        T.ours_time_mean(i), ...
        T.ours_rel_mean(i),   ...
        T.their_time_mean(i),  ...
        T.their_rel_mean(i));
end


% results = []; % Initialize an empty array of structs
% 
% for i_ = 1:numel(m_vals)
%     for j_ = 1:numel(r_vals)
%         m = m_vals(i_);
%         r = r_vals(j_);
% 
%         % You can extract specific fields from outputs if needed, for now just store them
%         results(end+1).m = m;
%         results(end).r = r;
%         results(end).ours_time = max(outputs_list_ours{i_, j_}.total_time);
%         results(end).ours_rel = outputs_list_ours{i_, j_}.relres(find(outputs_list_ours{i_, j_}.relres ~= 0, 1, 'last'));
%         results(end).their_time = max(cumsum(outputs_list_theirs{i_, j_}.time));
%         results(end).their_rel = outputs_list_theirs{i_, j_}.relres(find(outputs_list_theirs{i_, j_}.relres ~= 0, 1, 'last'));
%     end
% end

% % Convert to table
% T = struct2table(results);
% 
% % Display nicely
% disp(T)
% 
% for i = 1:height(T)
%     fprintf('%d & %d & \\num{%.4f} & \\num{%.2e} & \\num{%.4f} & \\num{%.2e} \\\\\n', ...
%         T.m(i), T.r(i), ...
%         T.ours_time(i), T.ours_rel(i), ...
%         T.their_time(i), T.their_rel(i));
% end