% N=1005;
data = load('email-Eu-core.txt');
src = data(:,1) + 1;
dst = data(:,2) + 1;
gnd= load('email-Eu-core-department-labels.txt');
gnd(:,2) = gnd(:,2) + 1;
% Create adjacency matrix (assuming undirected graph)
% A = sparse(src, dst, 1, N, N);
% A = A + A.';   % make it symmetric (if undirected)
% spy(A)
G = graph(src,dst); plot(G)
G = simplify(G);
A = adjacency(G,'weighted');

%%
r = 42;
a = tic;
V = A;
distances = pdist2(V, V, 'euclidean');
n = size(V, 1);
sigma = 35;  % adjust this parameter based on your data
W = exp(- distances.^2 / (2*sigma^2));
k = 15;  % choose appropriate k
for i=1:n
    [~, idx] = sort(distances(i, :), 'ascend');
    neighbors = idx(2:k+1);  % ignore the first one (distance to itself)
    mask = true(1, n);
    mask(neighbors) = false;
    W(i, mask) = 0;
end
V = (W + W') / 2;
% [W,H,iter,elapse,HIS] = NMF_SNMPBB(V,r);toc(a);
%% --- Replace Graph_SNMPBB with SVD-based factorization ---
a = tic;
[U,S,V] = svds(A, r);
W = abs(U * sqrt(S));
H = abs(sqrt(S) * V');
elapsed_time = toc(a);
fprintf('SVD factorization completed in %.2f seconds.\n', elapsed_time);

[W,H,output,acc] = Graph_SNMPBB(A,r,'knn',55,'w_init',W,'h_init',H,...
    'sym_weight',200,'graph_reg',10,'TRUELABEL',gnd(:,2));toc(a);

params.truelabel = gnd(:,2);
params.hinit = W;
[W, output, acc2] = symnmf_anls(A, r, params);

plot(acc); hold on; plot(acc2); hold off
%%
% [H_best, output_best, acc2] = symnmf_cluster(A, r, options);

% Determine number of clusters
num_clusters = size(W, 2);

% Assign colors to clusters
colors = lines(num_clusters);  % Or use 'jet', 'parula', etc.

% Get cluster assignments
[~, node_clusters] = max(W, [], 2);

% Plot the graph
figure;
h = plot(G, 'Layout', 'force');

% Color nodes by cluster
for i = 1:numnodes(G)
    highlight(h, i, 'NodeColor', colors(node_clusters(i), :), 'MarkerSize', 6);
end

title('Graph with Nodes Colored by Cluster');

figure

h = plot(G, 'Layout', 'force');

% Color nodes by cluster
colors = lines(num_clusters);
for i = 1:numnodes(G)
    g = gnd(i,2)
    c = colors(g,:);
    highlight(h, i, 'NodeColor', colors(gnd(i,2), :), 'MarkerSize', 6);
end

title('Graph with Nodes Colored by Cluster');
