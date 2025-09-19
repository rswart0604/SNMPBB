% Efficient EDVW -> P -> T (words = hyperedges, docs = vertices)

X = readlines("X_large.txt");
numDocs = numel(X);

%% Step 2: Tokenize manually (basic)
docsTokens = cell(numDocs,1);
allTokens = strings(0,1);

for d = 1:numDocs
    words = lower(split(X(d)));
    words = regexprep(words, '[^a-z0-9]', ''); % keep alphanum
    words = words(~cellfun('isempty', words));
    docsTokens{d} = words;
end
allTokens = vertcat(docsTokens{:});


vocab = unique(allTokens);
numWords = numel(vocab);
wordIndex = containers.Map(vocab, 1:numWords);

%% Step 3: Build sparse term frequency matrix
rows = []; cols = []; vals = [];

for d = 1:numDocs
    tokens = docsTokens{d};
    [u,~,idx] = unique(tokens);
    counts = accumarray(idx, 1);
    for j = 1:numel(u)
        rows(end+1) = d; %#ok<AGROW>
        cols(end+1) = wordIndex(u{j}); %#ok<AGROW>
        vals(end+1) = counts(j); %#ok<AGROW>
    end
end

TF = sparse(rows, cols, vals, numDocs, numWords);
save('TF.mat','TF','-v7.3')


%% --- assume TF (numDocs x numWords) exists (sparse) ---
% If not, produce TF as you already do, then:
% TF = sparse(rows, cols, vals, numDocs, numWords);

numDocs = size(TF,1);
numWords = size(TF,2);

%% TF-IDF (sparse)
df = sum(TF>0,1)';                      % numWords x 1 (column)
idf = log(double(numDocs) ./ (df + 1)); % column vector
% Apply idf: TF is docs x words, we want R = words x docs (E x V)
R = (TF .* (idf')).';                   % now E x V (words x docs), sparse

%% Pruning (sparsity param)
wordDocCount = sum(R>0,2);              % E x 1
keepMask = (wordDocCount > 1) & (wordDocCount < 0.5 * numDocs);
R = R(keepMask, :);                     % prune low/high frequency words
E = size(R,1);
V = size(R,2);
save('R.mat','R','-7.3')

%% Binary incidence (sparse)
Xbin = spones(R);                       % E x V sparse (word incidence)

%% Hyperedge weights (we) and hyperedge degrees (de)
% % Use population std (divide by N)
% we = std(R, 1, 2);                      % E x 1
% we(we == 0) = eps;                      % floor tiny weights
% de = sum(R, 2);                         % E x 1 (row sums)
% de(de == 0) = eps;
% todo check if this matches with smaller wos dataset
V = size(R,2);

rowSum = sum(R, 2);                  % sum over docs, sparse-safe
rowSumSq = sum(R.^2, 2);             % sum of squares, sparse-safe

meanRow = rowSum / V;                % E x 1
meanSqRow = rowSumSq / V;            % E x 1

varRow = meanSqRow - meanRow.^2;     % E x 1
varRow(varRow < 0) = 0;              % clip negatives (numerical noise)

we = sqrt(varRow);                   % E x 1, row std
we(we == 0) = eps;

de = rowSum;
de(de == 0) = eps;


%% Build M = (we ./ de) .* R  -------------- row-scale R
scales = we ./ de;                      % E x 1
M = spdiags(scales, 0, E, E) * R;       % E x V, sparse (row-scaled R)

%% Compute DV (vertex degree diag entries)  DV_vec = Xbin' * we
DV_vec = Xbin' * we;                    % V x 1
DV_vec(DV_vec == 0) = eps;

%% Compute S = W*D_E^{-1}*R == Xbin' * M  (V x V sparse)
S = full(Xbin' * M);                          % V x V, sparse

%% Compute P = D_V^{-1} * S  (row-normalize by DV_vec)
% Do row-scaling with sparse diagonal (cheap). Avoid dense diag.
P = spdiags(1./DV_vec, 0, V, V) * S;    % V x V, sparse
clear S

%% do weird thing
% Function that multiplies P with a vector x
opts.tol = 1e-6;
opts.maxit = 500;
fcn = @(x) P_times_x(x, Xbin, M, DV_vec);
[vecs, vals] = eigs(fcn, V, 1, 'lm', opts);



%% Stationary distribution pi (left eigenvector of P)
opts.tol = 1e-6;
opts.maxit = 500;
% compute dominant eigenvector of P' (largest magnitude)
[vecs, vals] = eigs(P', 1, 'lm', opts);
%%
pi = real(vecs);
pi = pi / sum(pi);                      % normalize to sum 1
pi = max(pi, eps);                      % guard against zeros


%%
sqrt_pi = sqrt(pi);
inv_sqrt_pi = 1 ./ sqrt_pi;

% M1 * x
M1_times_x = @(x) sqrt_pi .* P_times_x(inv_sqrt_pi .* x, Xbin, M, DV_vec);

% M2 * x
M2_times_x = @(x) inv_sqrt_pi .* P_transpose_times_x(sqrt_pi .* x, Xbin, M, DV_vec);
T_times_x = @(x) 0.5 * (M1_times_x(x) + M2_times_x(x));
[V_eigs, D_eigs] = eigs(T_times_x, V, k, 'lm', opts);



%% Build T efficiently (no dense diagonals)
sqrt_pi = sqrt(pi);
inv_sqrt_pi = 1 ./ sqrt_pi;

% M1 = Phi^{1/2} * P * Phi^{-1/2}
% Do left scale by sqrt_pi and right scale by inv_sqrt_pi:
M1 = spdiags(sqrt_pi, 0, V, V) * P;     % left-scale rows
M1 = bsxfun(@times, M1, inv_sqrt_pi');  % scale columns (sparse .* vector)

% M2 = Phi^{-1/2} * P' * Phi^{1/2}
M2 = spdiags(inv_sqrt_pi, 0, V, V) * P';
M2 = bsxfun(@times, M2, sqrt_pi');

% Symmetric T
T = 0.5 * (M1 + M2);                    % V x V sparse-ish
T = max(T, 0);                          % clip negatives
T = T - spdiags(diag(T), 0, V, V);      % zero diagonal
% save('T_large.mat','T','-v7.3')
% clear;clc;


%%
function y = P_times_x(x, Xbin, M, DV_vec)
    tmp = M * x;             % E × 1
    tmp2 = Xbin' * tmp;      % V × 1
    y = tmp2 ./ DV_vec;      % elementwise division
end

function y = P_transpose_times_x(x, Xbin, M, DV_vec)
    tmp  = Xbin * (x ./ DV_vec);  % E × 1
    y    = M' * tmp;              % V × 1
end


