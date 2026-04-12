clear;clc
X = load('3000_bullseye.data');
gnd = [ones(1000,1); 2*ones(1000,1); 3*ones(1000,1)]; k = 80;
r = 3;
num_runs = 20;

V = X;
[m, ~] = size(V);
distances = pdist2(V, V, 'euclidean');
sigma = mean(distances(:))*40;
W = exp(- distances.^2 / (2*sigma^2));
for i = 1:m
    [~, idx] = sort(distances(i, :), 'ascend');
    neighbors = idx(2:k+1);
    mask = true(1, m);
    mask(neighbors) = false;
    W(i, mask) = 0;
end
A = (W + W') / 2;

W_gsnmpbb_save = {};
W_snmpbb_save = {};
acc_gsnmpbb_save = zeros(num_runs,1);
acc_snmpbb_save = zeros(num_runs,1);

for i=1:num_runs
    [W_gsnmpbb,H_gsnmpbb,output_graphsnmpbb,acc_graphsnmpbb] = Graph_SNMPBB(A,r,'truelabel',gnd,'do_preprocess',false);
    W_gsnmpbb_save{i} = W_gsnmpbb;
    acc_gsnmpbb_save(i) = acc_graphsnmpbb(end);
    [W_snmpbb,H_snmpbb,output_snmpbb,acc_snmpbb] = NMF_SNMPBB(A,r,'truelabel',gnd);
    W_snmpbb_save{i} = W_snmpbb;
    acc_snmpbb_save(i) = acc_snmpbb(end);
end

[~, s_index] = min(abs(acc_snmpbb_save - mean(acc_snmpbb_save)));
[~, gs_index] = min(abs(acc_gsnmpbb_save - mean(acc_gsnmpbb_save)));

color = 'grb'; 
point = '.xo';

avg_W_s = W_snmpbb_save{s_index};
[~, cluster_labels] = max(avg_W_s, [], 2);
figure;
hold on;
for i = 1:r
    cluster_points = X(cluster_labels == i, :); % Extract points assigned to cluster i
    plot(cluster_points(:, 1), cluster_points(:, 2), [color(i), point(i)], 'MarkerSize', 8);
    writematrix(cluster_points, "s_cluster_" + i);
end
axis equal;
hold off;

avg_W_g = W_gsnmpbb_save{gs_index};
[~, cluster_labels] = max(avg_W_g, [], 2);
figure;
hold on;
for i = 1:r
    cluster_points = X(cluster_labels == i, :); % Extract points assigned to cluster i
    plot(cluster_points(:, 1), cluster_points(:, 2), [color(i), point(i)], 'MarkerSize', 8);
    writematrix(cluster_points, "gs_cluster_" + i);
end
axis equal;
hold off;