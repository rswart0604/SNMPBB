function [H, output, acc] = symnmf_newton(A, k, params)
%SYMNMF_NEWTON Newton-like algorithm for Symmetric NMF (SymNMF)
%   [H, output] = symnmf_newton(A, k, params) optimizes
%   the following formulation:
%
%   min_H f(H) = ||A - HH'||_F^2 subject to H >= 0
%
%   where A is a n*n symmetric matrix,
%         H is a n*k nonnegative matrix.
%         (typically, k << n)
%   'symnmf_newton' returns:
%       H: The low-rank n*k matrix used to approximate A.
%       output: Struct containing
%           - iter: number of iterations before termination
%           - obj: objective value f(H) at final solution H
%           - r:   relative error history
%           - ff:  Frobenius error history
%           - t:   elapsed time history
%
%   Parameters in 'params':
%       Hinit, maxiter, tol, sigma, beta, computeobj, debug
%

%% --- Input Checks ---
n = size(A, 1);
if n ~= size(A, 2)
    error('A must be a symmetric matrix!');
end

%% --- Parameters ---
if ~exist('params', 'var')
    H = 2 * full(sqrt(mean(mean(A)) / k)) * rand(n, k);
    maxiter = 100;
    tol = 1e-4;
    sigma = 0.1;
    beta = 0.1;
    computeobj = true;
    debug = 0;
    truelabel = [];
else
    if isfield(params, 'Hinit')
        [nH, kH] = size(params.Hinit);
        if nH ~= n, error('A and params.Hinit must have same number of rows!'); end
        if kH ~= k, error('params.Hinit must have k columns!'); end
        H = params.Hinit;
    else
        H = 2 * full(sqrt(mean(mean(A)) / k)) * rand(n, k);
    end
    maxiter    = get_option(params, 'maxiter',    100);
    tol        = get_option(params, 'tol',        1e-4);
    sigma      = get_option(params, 'sigma',      0.1);
    beta       = get_option(params, 'beta',       0.1);
    computeobj = get_option(params, 'computeobj', true);
    debug      = get_option(params, 'debug',      0);
    truelabel  = get_option(params, 'truelabel',  []);
end

%% --- Initialization ---
init_tic = tic;
projnorm_idx = false(n, k);
R = cell(1, k);
p = zeros(1, k);
left = H'*H;
obj = norm(A, 'fro')^2 - 2 * trace(H' * (A*H)) + trace(left * left);
gradH = 4 * (H * (H'*H) - A*H);
initgrad = norm(gradH, 'fro');
acc = [];

if debug
    fprintf('init grad norm %g\n', initgrad);
end

% Output struct
output.time(1) = toc(init_tic);
normA = norm(A,'fro');
relres_tic = tic;
XH = A*H;
cres = efficient_GetRes(normA,A,H,H,'XH',XH);
output.relres(1) = cres;
if ~isempty(truelabel)
    [~, est_label0] = max(H,[],2);
    tempacc = ClusteringMeasure(truelabel, est_label0);
    acc(1) = tempacc(1);
end
output.relres_time(1) = toc(relres_tic);


%% --- Main Loop ---
start_tic = tic;
for iter = 1:maxiter
    iter_tic = tic;

    gradH = 4*(H*(H'*H) - A*H);
    projnorm_idx_prev = projnorm_idx;
    projnorm_idx = gradH<=eps | H>eps;
    projnorm = norm(gradH(projnorm_idx));
    
    if projnorm < tol * initgrad || toc(start_tic) > 10
        if debug, fprintf('final grad norm %g\n', projnorm); end
        break;
    else
        if debug > 1, fprintf('iter %d: grad norm %g\n', iter, projnorm); end
    end
    
    if mod(iter, 100) == 0
        p = ones(1, k);
    end
      
    step = zeros(n, k);
    temp = H*H' - A;
    
    for i = 1:k
        if ~isempty(find(projnorm_idx_prev(:, i) ~= projnorm_idx(:, i), 1))
            hessian_i = hessian_blkdiag(temp, H, i, projnorm_idx);
            [R{i}, p(i)] = chol(hessian_i);
        end
        if p(i) > 0
            step(:, i) = gradH(:, i);
        else
            step_temp = R{i}' \ gradH(projnorm_idx(:, i), i);
            step_temp = R{i} \ step_temp;
            step_part = zeros(n, 1);
            step_part(projnorm_idx(:, i)) = step_temp;
            step_part(step_part > -eps & H(:, i) <= eps) = 0;
            if sum(gradH(:, i) .* step_part) / norm(gradH(:, i)) / norm(step_part) <= eps
                p(i) = 1;
                step(:, i) = gradH(:, i);
            else
                step(:, i) = step_part;
            end
        end
    end
    
    % Armijo rule
    alpha_newton = 1;
    while true
        Hn = max(H - alpha_newton * step, 0);
        left = Hn'*Hn;
        newobj = norm(A, 'fro')^2 - 2 * trace(Hn' * (A*Hn)) + trace(left * left);
        if newobj - obj <= sigma * sum(sum(gradH .* (Hn-H)))
            H = Hn;
            obj = newobj;
            break;
        else
            alpha_newton = alpha_newton * beta;
        end
    end
    
    % Update history
    output.time(iter+1) = toc(iter_tic);
    relres_tic = tic;
    XH = A*H;
    cres = efficient_GetRes(normA,A,H,H,'XH',XH);
    output.relres(1) = cres;
    if ~isempty(truelabel)
        [~, est_label0] = max(H,[],2);
        tempacc = ClusteringMeasure(truelabel, est_label0);
        acc(iter+1) = tempacc(1);
    end
    output.relres_time(1) = toc(relres_tic);    
end

output.total_time = cumsum(output.time);

%% --- Finalize ---
output.iter = iter;
if computeobj
    output.obj = obj;
else
    output.obj = -1;
end

end % function


%----------------------------------------------------
function He = hessian_blkdiag(temp, H, idx, projnorm_idx)
[n, ~] = size(H);
subset = find(projnorm_idx(:, idx) ~= 0); 
hidx = H(subset, idx);
eye0 = (H(:, idx)' * H(:, idx)) * eye(n);
He = 4 * (temp(subset, subset) + hidx * hidx' + eye0(subset, subset));
end

%----------------------------------------------------
function val = get_option(params, field, default)
if isfield(params, field)
    val = params.(field);
else
    val = default;
end
end
