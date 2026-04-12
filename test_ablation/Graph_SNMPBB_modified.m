function [W,H,output,acc]=Graph_SNMPBB_modified(V,r,varargin)

V_original = V;

if ~exist('V','var'), error('please input the sample matrix.'); end
if ~exist('r','var'), error('please input the low rank.'); end

[m,~] = size(V);

start_tic = tic;

% ---------------- Default settings ----------------
MaxIter = 3000;
MinIter = 500;
MaxTime = 30;
tol = 1e-7;
verbose = 0;


% default knn
if m > 1000
    num_knn = 50;
else
    num_knn = max(5, round(5*2^((log(m)/log(3) - 4))));
end

options.bb = true;
options.nonmonotone = true;
options.second_descent_step = true;
options.s = false;

sym_weight = 0; graph_reg = 0;

% parse optional args (name-value pairs)
ITER_MAX = 1000;
do_preprocess = false;
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
        case 'INNER_MAX_ITER', ITER_MAX=varargin{i+1};
        case 'GRAPH_REG',  graph_reg = val;
        case 'KNN',        num_knn = val;
        case 'TRUELABEL',  truelabel = val;
        case 'SIGMA',      sigma = val;
        case 'DO_PREPROCESS', do_preprocess = val;
        case "SECOND_DESCENT_STEP",    options.second_descent_step = val;
        case "BB", options.bb = val;
        case "NONMONOTONE", options.nonmonotone = val;
        case "S", options.s = val;
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

    sigma = mean(distances(:))*40;    
    
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
if sym_weight == 0
    sym_weight = 2 * norm(A, 'fro') / sqrt(m * r);
end
if graph_reg == 0
    graph_reg = 4*sym_weight;
end
if graph_reg == -1
    graph_reg = 0;
end



d = 1 ./ sqrt(sum(A,2) + eps);
L = speye(m) - spdiags(d,0,m,m) * A * spdiags(d,0,m,m);
V = sparse(A);
L = sparse(L);


% inner solver parameters
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
normX = norm(V_original, 'fro');
XH = V_original*H';
cres = efficient_GetRes(normX,V_original,W,H','XH',XH);
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
    [W, iterW, GradW] = SNMPBB_(W, HHt, HVt, ITER_MAX, ITER_MIN, tolW, 0, [], options); 
    if iterW <= ITER_MIN
        tolW = tolW / 10;
    end
    
    % --- Update H (H is r x n) with graph regularization ---
    WtW = W * W' + sym_weight * eye(r);            % r x r (since W is r x m)
    WtV = (V * W' + sym_weight * W')';             % r x n
    [H, iterH, GradH] = SNMPBB_(H, WtW, WtV, ITER_MAX, ITER_MIN, tolH, graph_reg, L, options);
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
    % normX = norm(V, 'fro');
    XH = V_original*H';
    cres = efficient_GetRes(normX,V_original,W',H','XH',XH);
    % if options.second_descent_step == 0
    %     disp(cres)
    % end
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
function [x, iter, gradx] = SNMPBB_(x0, WtW, WtV, iterMax, iter_Min, tol, graph_reg_local, L_local, options)
    % SNMPBB: Quadratic regularization projected Barzilai--Borwein method:
    %   min 1/2 * <x, WtW*x> - <x, WtV> + (graph_reg/2) tr(x*L*x') subject to x>=0
    if options.s
        s = options.s;
    else
        s = 1;
    end
    eta = 0.75;
    lamax = 1e5; lamin = 1e-20;
    gamma = 1e-4;
    rho = 0.25;
    
    x = x0;
    delta0 = -sum(sum(x .* WtV));
    dQd0 = sum(sum((WtW * x) .* x));
    fn = delta0 + 0.5 * dQd0;
    lambda_bb = 1;
    
    Lipschitz = 1 / max(eps, norm(full(WtW)));
    if graph_reg_local > 0
        Lipschitz = 1 / max(eps, norm(full(WtW)) + 3*graph_reg_local);
        % disp(norm(full(L_local)));
    end

    gradx = WtW * x - WtV;
    if graph_reg_local > 0
        gradx = gradx + 2 * graph_reg_local * (x * L_local);
    end
    
    iter = 1;
    if options.second_descent_step
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
            
            if options.nonmonotone

                % nonmonotone search preparation
                func(iter) = fn;
                if iter == 1
                    S(1) = fn;
                else
                    S(iter) = fn + eta * (S(iter-1) - fn);
                end
            else
                func(iter) = fn;
                S(iter) = fn;
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
            while (fn > S(iter) + alpha * gamma * (delta))% && (alpha > 1e-20)
                alpha = rho * alpha;
                fn = func(iter) + alpha * delta + 0.5 * alpha^2 * dQd;
            end
            x = x + s .* alpha .* dx;
            gradx = gradx + alpha .* dgradx;

            if options.bb
                sty = dQd;
                if sty > 0
                    sts = dx(:)' * dx(:);
                    lambda_bb = min(lamax, max(lamin, sts / sty));
                else
                    lambda_bb = lamax;
                end
            else
                lambda_bb = 1;
            end
            % if iter == 1
            %     eta_old = eta;
            %     eta = eta / 2;
            % else
            %     tmp = (eta + eta_old) / 2;
            %     eta_old = eta;
            %     eta = tmp;
            % end
        end % for iter
    else
        for iter = 1:iterMax
            if iter >= iter_Min
                pgn = norm(gradx(gradx < 0 | x > 0));
                if pgn <= tol
                    break;
                end
            end
            % first (safe) step only
            dx = max(x - Lipschitz .* gradx, 0) - x;
            dgradx = WtW * dx;
            if graph_reg_local > 0
                dgradx = dgradx + 2 * graph_reg_local * (dx * L_local);
            end
            x = x + dx;
            gradx = gradx + dgradx;
        end

        % iter = 2;
        % dx = max(x - Lipschitz .* gradx, 0) - x;
        % x = x + dx;
    end
    
    if iter == iterMax
        fprintf('Max iter in QRPBB\n');
    end
end % SNMPBB_

end
