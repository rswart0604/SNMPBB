function [H, output, acc] = symnmf_anls(A, k, params)
%SYMNMF_ANLS ANLS algorithm for SymNMF with structured output
%
%   [H, output] = symnmf_anls(A, k, params)
%
%   See original comments for details.

n = size(A, 1);
if n ~= size(A, 2)
    error('A must be a symmetric matrix!');
end

init_start = tic;
% Parse params
if ~exist('params', 'var')
    H = 2 * full(sqrt(mean(mean(A)) / k)) * rand(n, k);
    maxiter = 10000;
    tol = 1e-10;
    alpha = max(max(A))^2;
    computeobj = true;
    debug = 0;
    truelabel = [];
else
    if isfield(params, 'Hinit')
        [nH, kH] = size(params.Hinit);
        if nH ~= size(A, 1)
            error('A and params.Hinit must have same number of rows!');
        end
        if kH ~= k
            error('params.Hinit must have k columns!');
        end
        H = params.Hinit;
    else
        H = 2 * full(sqrt(mean(mean(A)) / k)) * rand(n, k);
    end
    if isfield(params, 'maxiter'), maxiter = params.maxiter; else, maxiter = 10000; end
    if isfield(params, 'tol'), tol = params.tol; else, tol = 1e-1; end
    if isfield(params, 'alpha') && params.alpha >= 0
        alpha = params.alpha;
    else
        alpha = max(max(A))^2;
    end
    if isfield(params, 'computeobj'), computeobj = params.computeobj; else, computeobj = true; end
    if isfield(params, 'debug'), debug = params.debug; else, debug = 0; end
    if isfield(params, 'truelabel'), truelabel = params.truelabel; else, truelabel = [];
end


% Initialization
W = H;
I_k = alpha * eye(k);
left = H' * H;
right = A * H;
acc = [];

% Output struct
output.relres = [];
output.obj = [];
output.time = [];
output.relres_time = [];
output.acc = [];

output.time(1) = toc(init_start);
normA = norm(A, 'fro');
res_time_start = tic;
output.obj(1) = norm(A-H*H','fro') + alpha*norm(W-H,'fro');
XH = A*H;
cres = efficient_GetRes(normA,A,W,H,'XH',XH);
output.relres(1) = cres;
output.relres_time(1) = toc(res_time_start);
if ~isempty(truelabel)
    [~, est_label0] = max(H,[],2);
    tempacc = ClusteringMeasure(truelabel, est_label0);
    acc(1) = tempacc(1);
end



% Main loop
for iter = 1:maxiter
    loop_tic = tic;

    % --- Update W ---
    W = nnlsm_blockpivot(left + I_k, (right + alpha * H)', 1, W')';
    left = W' * W;
    right = A * W;

    % --- Update H ---
    H = nnlsm_blockpivot(left + I_k, (right + alpha * W)', 1, H')';
    temp = alpha * (H - W);

    % Gradients for stopping criterion
    gradH = H * left - right + temp;
    left = H' * H;
    right = A * H;
    gradW = W * left - right - temp;

    % Record projected grad norm here so it's included in iter time
    projnorm = sqrt(norm(gradW(gradW<=0 | W>0))^2 + ...
                        norm(gradH(gradH<=0 | H>0))^2);

    % Record iteration time
    output.time(iter+1) = toc(loop_tic);

    % Compute residual & objective
%     if mod(iter, 3) == 0
    res_time_tic = tic;
    XH = A*H;
    cres = efficient_GetRes(normA,A,W,H,'XH',XH);
    output.relres(iter+1) = cres;
    if ~isempty(truelabel)
        [~, est_label] = max(H,[],2);
        tempacc = ClusteringMeasure(truelabel, est_label);
        acc(iter+1) = tempacc(1);
    end
    output.obj(iter+1) = cres;
    output.relres_time(iter+1) = toc(res_time_tic);
%     end

    % Check stopping condition
    if iter == 1
        initgrad = sqrt(norm(gradW(gradW<=0 | W>0))^2 + ...
                        norm(gradH(gradH<=0 | H>0))^2) / 100;
        if debug, fprintf('init grad norm %g\n', initgrad); end
    else
        % Use projnorm here
        if projnorm < tol * initgrad
            if debug, fprintf('final grad norm %g\n', projnorm); end
            break;
        elseif debug > 1
            fprintf('iter %d: grad norm %g\n', iter, projnorm);
        end
    end
end

if alpha == 0
    norms_W = sqrt(sum(W.^2));
    norms_H = sqrt(sum(H.^2));
    norms = sqrt(norms_W .* norms_H);
    W = bsxfun(@times, W, norms./norms_W);
    H = bsxfun(@times, H, norms./norms_H);
end

if computeobj
    output.final_obj = norm(A, 'fro')^2 - 2 * trace(W' * (A*H)) + trace((W'*W) * (H'*H));
else
    output.final_obj = -1;
end

output.total_time = cumsum(output.time);

end % function
