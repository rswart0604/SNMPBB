function [H_best, output_best, acc_best] = symnmf_cluster(X, k, options)
%SYMNMF_CLUSTER
%   [H_best, obj_best] = symnmf_cluster(X, k, options)
%
%   Thin wrapper for symmetric NMF clustering. Builds a similarity 
%   graph from X, then calls symnmf_anls or symnmf_newton. 
%   Returns only the best H and its objective value.

[n, ~] = size(X);

% === Defaults ===
if ~exist('options','var'), options = struct(); end
if ~isfield(options,'graph_type'),      options.graph_type = 'sparse'; end
if ~isfield(options,'similarity_type'), options.similarity_type = 'gaussian'; end
if ~isfield(options,'graph_objfun'),    options.graph_objfun = 'ncut'; end
if ~isfield(options,'kk'),              options.kk = floor(log2(n)) + 1; end
if ~isfield(options,'nn'),              options.nn = 7; end
if ~isfield(options,'tol'),             options.tol = 5e-4; end
if ~isfield(options,'maxiter'),         options.maxiter = 10000; end
if ~isfield(options,'rep'),             options.rep = 1; end
if ~isfield(options,'Hinit'),           options.Hinit = []; end
if ~isfield(options,'Winit'),           options.Winit = []; end
if ~isfield(options,'alg'),             options.alg = 'anls'; end
if ~isfield(options,'truelabel'),       options.truelabel = []; end
if ~isfield(options,'second_descent_step'),       options.second_descent_step = true; end
if ~isfield(options,'s'),       options.s = false; end
if ~isfield(options,'bb'),       options.bb = true; end
if ~isfield(options,'nonmonotone'),       options.nonmonotone = true; end

% === Build similarity matrix A ===
init_tic = tic;
D = dist2(X,X);
if strcmp(options.graph_type,'full') && strcmp(options.similarity_type,'gaussian')
    A = scale_dist3(D, options.nn);
elseif strcmp(options.graph_type,'full') && strcmp(options.similarity_type,'inner_product')
    A = X * X';
elseif strcmp(options.graph_type,'sparse') && strcmp(options.similarity_type,'gaussian')
    A = scale_dist3_knn(D, options.nn, options.kk, true);
else % sparse + inner_product
    Xnorm = X';
    d = 1 ./ sqrt(sum(Xnorm.^2));
    Xnorm = bsxfun(@times, Xnorm, d);
    A = inner_product_knn(D, Xnorm, options.kk, true);
    clear Xnorm d;
end
clear D;

if strcmp(options.graph_objfun,'ncut')
    dd = 1 ./ sum(A);
    dd = sqrt(dd);
    A = bsxfun(@times, A, dd);
    A = A';
    A = bsxfun(@times, A, dd);
    clear dd;
end
A = (A + A')/2;

% === Params for solver ===
params.maxiter   = options.maxiter;
params.tol       = options.tol;
params.truelabel = options.truelabel;

% === Run multiple reps and keep the best ===
obj_best = Inf;
H_best   = [];

init_time = toc(init_tic);

for i = 1:options.rep
    if ~isempty(options.Hinit)
        params.Hinit = options.Hinit(:,:,i);
    end
    if ~isempty(options.Winit)
        params.Winit = options.Winit(:,:,i);
    end

    if strcmp(options.alg,'newton')
        [H, output, acc] = symnmf_newton(A, k, params);
    elseif strcmp(options.alg,'graph_snmpbb')
        % for orl
        % sym_weight = 0.06;
        % graph_reg = 1.3;

        % for everything else
        sym_weight = 10;
        graph_reg = 3;

        % sym_weight = 0.05;
        % graph_reg = 1.2;

        % sym_weight = 0.09;
        % graph_reg = 1.1;

        % sym_weight = 0.07;
        % graph_reg = 1.3;
        if ~isempty(options.Hinit)
            [H,~,output,acc] = Graph_SNMPBB(A,k,'truelabel',params.truelabel,'do_preprocess',false,...
                'sym_weight',sym_weight,'graph_reg',graph_reg,'W_INIT',params.Hinit,'H_INIT',params.Winit);
        else
            [H,~,output,acc] = Graph_SNMPBB(A,k,'truelabel',params.truelabel,'do_preprocess',false);
        end

    elseif strcmp(options.alg,'modified_graph_snmpbb')
        % sym_weight = 0.3;
        % graph_reg = .5;
        [H,~,output,acc] = Graph_SNMPBB_modified(A,k,'truelabel',params.truelabel,'do_preprocess',false, ...
            'W_INIT',params.Hinit,'H_INIT',params.Winit, ...
            'do_preprocess', false,'second_descent_step',options.second_descent_step, 'bb', options.bb, ...
            'nonmonotone', options.nonmonotone, 's', options.s);

    elseif strcmp(options.alg,'modified_pgd')
        % use the higher scaling
        [H,output,acc] = PGD_modified(A,k,'TRUELABEL',params.truelabel,'U_INIT',params.Winit');
    elseif strcmp(options.alg,'pgd')
        [H,output,acc] = PGD(A,k,'TRUELABEL',params.truelabel,'U_INIT',params.Winit');
    else
        [H, output, acc] = symnmf_anls(A, k, params);
    end

    obj = output.relres(end); % final objective value
    if obj < obj_best
        obj_best = obj;
        output_best = output;
        output_best.total_time = output_best.total_time + init_time;
        output_best.time(1) = output_best.time(1) + init_time;
        H_best   = H;
        acc_best = acc;
    end
end
