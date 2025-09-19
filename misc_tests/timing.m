clear; clc;

m_vals = [200, 400, 500];
r_vals = [1,2,4,8,16,32,64];
num_runs = 1;


for i_=1:numel(m_vals)
    for j_=1:numel(r_vals)
        m = m_vals(i_);
        r = r_vals(j_);
        for i = 1:num_runs
            W_old = max(0, randn(m, r));
            V = W_old * W_old';
            [W,H,output,~] = NMF_SNMPBB(V, r);
            outputs_list{i_,j_} = output;
        end    
    end
end
    
for i=1:6
    avg_time(i) = mean(outputs_list{2,i}.time(2:end));
    if i > 1
        increase(i-1) = avg_time(i)/avg_time(i-1);
    end
end
