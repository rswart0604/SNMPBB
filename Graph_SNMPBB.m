function [W,H,output,acc]=Graph_SNMPBB(V,r,varargin)
% Graph_SNMPBB ::: Refactored Graph-regularized SNMPBB NMF that returns
% [W, H, output, acc].
%
% Usage:
%   [W,H,output,acc] = new_graph_nmf(V,r,'GRAPH_REG',alpha,'KNN',k,'TRUELABEL',labels,...)
%
% output fields:
%   output.relres      - relative residual history
%   output.time        - per-iteration elapsed times
%   output.relres_time - time spent computing residual and cluster acc
%   output.total_time  - cumulative times
%
% acc : clustering accuracy history if 'TRUELABEL' provided, otherwise 0

if ~exist('V','var'), error('please input the sample matrix.'); end
if ~exist('r','var'), error('please input the low rank.'); end

[m,~] = size(V);

start_tic = tic;

% ---------------- Default settings ----------------
MaxIter = 3000;
MinIter = 500;
MaxTime = 100000;
tol = 1e-7;
verbose = 0;
% sym_weight = mean(V(:).^2) * .0001
% graph_reg = .00000001 * norm(V,'fro')^2 / m

% for ORL, sym 1.2 and graph 1.
% sym_weight = 1.2;
% graph_reg = 1.5;

% sym_weight = 3;
% graph_reg = 4;

% this is good for COIL20!!!!
% graph_reg = 0.01 * norm(V, 'fro')
% sym_weight = 0.1 * (m/10)/(log(r)/log(5))           % parameter for weight of symmetric penalty
% graph_reg = 2;
% sym_weight = 3.5;
graph_reg = .5;
sym_weight = .5;


% default knn
if m > 1000
    num_knn = 50;
else
    num_knn = max(5, round(5*2^((log(m)/log(3) - 4))));
end
num_knn = 80;

% parse optional args (name-value pairs)
do_preprocess = true;
truelabel = [];
w_init = false; h_init = false;
for i = 1:2:length(varargin)
    if i+1 > length(varargin), error('Optional parameters should always go by pairs'); end
    name = upper(varargin{i});
    val  = varargin{i+1};
    switch name
        case 'MAX_ITER',   MaxIter = val;
        case 'MIN_ITER',   MinIter = val;
        case 'MAX_TIME',   MaxTime = val;
        case 'W_INIT',     W0 = val; w_init = true;
        case 'H_INIT',     H0 = val; h_init = true;
        case 'TOL',        tol = val;
        case 'VERBOSE',    verbose = val;
        case 'SYM_WEIGHT', sym_weight = val;
        case 'GRAPH_REG',  graph_reg = val;
        case 'KNN',        num_knn = val;
        case 'TRUELABEL',  truelabel = val;
        case 'SIGMA',      sigma = val;
        case 'DO_PREPROCESS', do_preprocess = val;
        otherwise
            error('Unrecognized option: %s', varargin{i});
    end
end


if do_preprocess
    % compute pairwise distances (m x m)
    distances = pdist2(V, V, 'euclidean');
    
    % default sigma if not provided
    if ~exist('sigma','var')
        sigma = 35;
    end
    % For k-nearest neighbor graph, zero out entries not among the k-nearest neighbors:
    
    % k = 3 for ORL right now gives 78 accuracy?
    % k = 12 for COIL20 is best
    % k = 42 for MNIST train is best
    % k = 40 for Isolet1 is best
    
    % kk = floor(log2(m)) + 1;
    % D = dist2(V,V);
    % V_ = scale_dist3_knn(D, 7, kk, true);

    sigma = mean(distances(:))*40;
    % sigma = 35;
    
    
    W = exp(- distances.^2 / (2*sigma^2));
    
    k = num_knn;  % choose appropriate k
    for i = 1:m
        [~, idx] = sort(distances(i, :), 'ascend');
        neighbors = idx(2:k+1);  % ignore the first one (distance to itself)
        mask = true(1, m);
        mask(neighbors) = false;
        W(i, mask) = 0;
    end
    
    % Symmetrize affinity
    A = (W + W') / 2;
else
    A = V;
end

d = 1 ./ sqrt(sum(A,2) + eps);
L = speye(m) - spdiags(d,0,m,m) * A * spdiags(d,0,m,m);
V = sparse(A);
L = sparse(L);


% inner solver parameters
ITER_MAX = 1000;
ITER_MIN = 1;

% Precompute a few things for stopping criteria
if w_init || h_init
    scale = sqrt(mean(mean(A)) / r) / mean(mean(W0));
end
if w_init
    W = W0 * scale;
else
    W = 2 * full(sqrt(mean(mean(V)) / r)) * rand(m, r);
end
if h_init
    H = H0 * scale;
else
    H = 2 * full(sqrt(mean(mean(V)) / r)) * rand(r,m);
end
HHt = H * H' + sym_weight * eye(r);
HVt = (V * H' + sym_weight * H')';
WtW = W' * W + sym_weight * eye(r);
WtV = (V * W + sym_weight * W)';
GradW = HHt * W' - HVt;
GradH = WtW * H - WtV;
if graph_reg > 0
    GradH = GradH + 2 * graph_reg * (H * L);
end
init_delta = norm([GradW; GradH],'fro');
tolH = max(tol,1e-3) * init_delta;
tolW = tolH;
constV = sum(sum(V.^2));

% ---------------- Histories ----------------
e_hist = []; t_hist = []; relres_time_hist = [];
measure_time = 0;  % time spent computing clustering measure each iter
etime = tic;

% initial records
t_hist(1) = toc(start_tic);
res_tic = tic;
normX = norm(V, 'fro');
XH = V*H';
cres = efficient_GetRes(normX,V,H',H','XH',XH);
e_hist(1) = cres;
relres_time_hist(1) = toc(res_tic);

if ~isempty(truelabel)
    [~, label0] = max(W, [], 2);
    tempacc = ClusteringMeasure(truelabel, label0);
    acc = tempacc(1);
else
    acc = 0;
end

% we transpose W before calling SNMPBB_ (inner solver expects x shaped as r x m)
W = W';

% ---------------- Main loop ----------------
for iter = 1:MaxIter
    iter_tic = tic;
    
    % --- Update W (passed as r x m matrix into SNMPBB_) ---
    HHt = H * H' + sym_weight * eye(r);            % r x r
    HVt = (V * H' + sym_weight * H')';             % r x m
    [W, iterW, GradW] = SNMPBB_(W, HHt, HVt, ITER_MAX, ITER_MIN, tolW, V, 0, []); 
    if iterW <= ITER_MIN, tolW = tolW / 10; end
    
    % --- Update H (H is r x n) with graph regularization ---
    WtW = W * W' + sym_weight * eye(r);            % r x r (since W is r x m)
    WtV = (V * W' + sym_weight * W')';             % r x n
    [H, iterH, GradH] = SNMPBB_(H, WtW, WtV, ITER_MAX, ITER_MIN, tolH, V, graph_reg, L);
    if iterH <= ITER_MIN, tolH = tolH / 10; end

    % compute projected gradient measure and histories
    delta = norm([GradW(GradW < 0 | W > 0); GradH(GradH < 0 | H > 0)]);
        
    % compute time for iteration excluding clustering measurement
    t_hist(end+1) = toc(iter_tic);
    
    measure_tic = tic;
    % clustering accuracy (if provided) and record time used to compute it
    if ~isempty(truelabel)
        [~, label] = max(H, [], 1);
        tempacc = ClusteringMeasure(truelabel, label);
        acc(end+1) = tempacc(1);
    end
    normX = norm(V, 'fro');
    XH = V*H';
    cres = efficient_GetRes(normX,V,H',H','XH',XH);
    e_hist(end+1) = cres;
    relres_time_hist(end+1) = toc(measure_tic);
    
    % stopping criteria
    if (delta <= tol * init_delta && iter >= MinIter) || t_hist(end) >= MaxTime
        break;
    end
    if iter > 1
        % relative change of error
        if abs(e_hist(end) - e_hist(end-1)) / max(e_hist(end-1), eps) < 1e-7
            break;
        end
    end
    
    % optionally print
    if verbose == 2 && rem(iter,10) == 0
        fprintf('%d:\tstopping criteria = %e,\tobjective value = %e.\n', iter, delta/init_delta, e_hist(end) + constV);
    end
end

% final transpose of W back to m x r
W = W';

elapse = toc(etime);

% ---------------- Pack output struct ----------------
output.relres = e_hist;
output.time = t_hist;
output.relres_time = relres_time_hist;
output.total_time = cumsum(output.time);

% if no acc history (truelabel not given) ensure acc is 0
if isempty(acc), acc = 0; end

% ---------------- Nested function: SNMPBB_ (unchanged, but local) ----------------
function [x, iter, gradx] = SNMPBB_(x0, WtW, WtV, iterMax, iter_Min, tol, V_in, graph_reg_local, L_local)
    % SNMPBB: Quadratic regularization projected Barzilai--Borwein method for NNLS:
    %   min 1/2 * <x, WtW*x> - <x, WtV> + (graph_reg/2) tr(x*L*x') subject to x>=0
    s = 1.5;
    eta = 0.75;
    lamax = 1e5; lamin = 1e-20;
    gamma = 1e-4;
    rho = 0.25;
    
    x = x0;
    delta0 = -sum(sum(x .* WtV));
    dQd0 = sum(sum((WtW * x) .* x));
    fn = delta0 + 0.5 * dQd0;
    lambda_bb = 1;
    
    % Lipschitz estimate (safe)
    Lipschitz = 1 / max(eps, norm(full(WtW)));
    gradx = WtW * x - WtV;
    if graph_reg_local > 0
        % note: x * L_local makes sense when x is r x n and L_local is n x n,
        % or when x is r x m and L_local is m x m depending on the call.
        gradx = gradx + 2 * graph_reg_local * (x * L_local);
    end
    
    for iter = 1:iterMax
        if iter >= iter_Min
            pgn = norm(gradx(gradx < 0 | x > 0));
            if pgn <= tol
                break;
            end
        end
        
        % take a Lipschitz step
        dx = max(x - Lipschitz .* gradx, 0) - x;
        dgradx = WtW * dx;
        if graph_reg_local > 0
            dgradx = dgradx + 2 * graph_reg_local * (dx * L_local);
        end
        delta = dx(:)' * gradx(:);
        dQd = dx(:)' * dgradx(:);
        x = x + dx;
        fn = fn + delta + 0.5 * dQd;
        gradx = gradx + dgradx;
        
        % nonmonotone search preparation
        func(iter) = fn;
        if iter == 1
            S(1) = fn;
        else
            S(iter) = fn + eta * (S(iter-1) - fn);
        end
        
        dx = max(x - lambda_bb .* gradx, 0) - x;
        dgradx = WtW * dx;
        if graph_reg_local > 0
            dgradx = dgradx + 2 * graph_reg_local * (dx * L_local);
        end
        delta = dx(:)' * gradx(:);
        dQd = dx(:)' * dgradx(:);
        fn = func(iter) + delta + 0.5 * dQd;
        alpha = 1;
        while (fn > S(iter) + alpha * gamma * (delta))
            alpha = rho * alpha;
            fn = func(iter) + alpha * delta + 0.5 * alpha^2 * dQd;
        end
        x = x + s .* alpha .* dx;
        
        sty = dQd;
        gradx = gradx + alpha .* dgradx;
        if sty > 0
            sts = dx(:)' * dx(:);
            lambda_bb = min(lamax, max(lamin, sts / sty));
        else
            lambda_bb = lamax;
        end
        
        if iter == 1
            eta_old = eta;
            eta = eta / 2;
        else
            tmp = (eta + eta_old) / 2;
            eta_old = eta;
            eta = tmp;
        end
    end % for iter
    
    if iter == iterMax
        fprintf('Max iter in QRPBB\n');
    end
end % SNMPBB_

end
