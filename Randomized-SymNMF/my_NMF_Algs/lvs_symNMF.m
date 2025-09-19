function [W,output] = lvs_symNMF(X,k,varargin)
%% SymNMF using leverage score sampling to solve the nonnegative least squares problem
% Here are some more notes on the parameters:
% 
% 1) hybrid_sample : toggle for hybrid sampling
% 2) topk_sample : will only sample a certain topk number of rows with the
% highest 'topk' leverage scores. This did not work well in my experience.
    %% Parse inputs
    inParams = inputParser;
    
    addParameter(inParams,'max_iters',50);
    addParameter(inParams,'min_iters',10);
    addParameter(inParams,'print_on',false);
    addParameter(inParams,'init',{});
    addParameter(inParams,'sfrac',0.1);
    addParameter(inParams,'hybrid_sample',false);
    addParameter(inParams,'topk_sample',false);
    addParameter(inParams,'alg_str','hals'); % update rule to use
    addParameter(inParams,'alpha',-1);
    addParameter(inParams,'track_e',true);
    addParameter(inParams,'use_stable_lev',false); % use householder qr if true for leverage score computation
    addParameter(inParams,'tol',1e-5);
    addParameter(inParams,'maxtime',Inf);
    addParameter(inParams,'prjG',false);
    addParameter(inParams,'tau',1.0);
    addParameter(inParams,'track_theta',true); % track fraction of deterministically sampled leverage score mass
    
    parse(inParams,varargin{:});
    
    max_iters   = inParams.Results.max_iters;
    min_iters   = inParams.Results.min_iters;
    print_on    = inParams.Results.print_on;
    init        = inParams.Results.init;
    sfrac       = inParams.Results.sfrac;
    hybrid_sample  = inParams.Results.hybrid_sample;
    topk_sample = inParams.Results.topk_sample;
    alpha       = inParams.Results.alpha;
    track_e     = inParams.Results.track_e;
    use_stable_lev        = inParams.Results.use_stable_lev;
    maxtime     = inParams.Results.maxtime;
    prjG        = inParams.Results.prjG;
    alg_str     = inParams.Results.alg_str;
    tau         = inParams.Results.tau;
    track_theta = inParams.Results.track_theta;
    %% Init some algorithm parameters
    [m,n] = size(X);
    if ~issymmetric(X)
        error('Input matrix X must by symmetric');
    end
    % error and time tracking
    output.relres   = zeros(1,max_iters+1);
    output.time     = zeros(1,max_iters+1);
    e_list          = zeros(2,max_iters);
    t_list          = zeros(4,max_iters);
    output.prj_grad = zeros(1,max_iters+1);
    output.matmul_time  = 0;
    output.solver_time  = 0;
    output.sampler_time = 0;
    
    if alpha < 0
        alpha = full(max(max(X))^2);
    end
    init_start = tic;
    % initialize factor matrices
    if isempty(init)
        H = 2 * full(sqrt(mean(mean(X)) / k)) * rand(n, k);
    else
        H = init.H;
    end
    W = H;
    normX   = norm(X,'fro');
    
    WtX = W'*X;
    cres    = efficient_GetRes(normX,X,H,H,'WtX',WtX);
    output.relres(1) = cres;
    if print_on
        fprintf('START, Relative residual : %f\n', output.relres(1));
    end
    s = ceil(sfrac*m);
    Xt = X';
    output.time(1) = output.time(1) + toc(init_start);
    
    % Compute the Projected Gradient
    if prjG
        output.prj_grad(1) = efficient_GetPrjGradNrm(X,W,W,'WtX',WtX,'tol',eps,'doSym',true);
    end
    
    %% Main Iteration
    for i = 1:max_iters
        loop_start = tic;

        %% Update using leverage score method
        %% Start update W
        sampler_start = tic;

        % Compute the Leverage scores
        if i == 1 && print_on
            fprintf('Doing full levScore Sampling\n')
        end
        if use_stable_lev % use regular QR or cholesky QR
            [lev_scoresH] = get_Full_LevScores(H);
        else
            [lev_scoresH] = get_Full_LevScores_wCholQR(H);
        end
        % Determine the sampling strategy
        if hybrid_sample
            if print_on && i==1
                fprintf('Doing Hybrid (Det) Sampling\n');
            end
           % partially deterministic sample
            [didx,C,D] = my_ProcLevSample_det(lev_scoresH,s,tau);
            sH = [H(didx,:);D*H(C,:)]; % row stack the deterministic samples and random samples
            sX = [X(:,didx),X(:,C)*D]; % col stack the sampled X values
            if track_theta
                t_list(1,i) = sum(lev_scoresH(C)); % sum of deterministic lvs scores
                t_list(3,i) = sum(numel(C)); % sum of deterministic lvs scores
            end
        elseif topk_sample
            % topk samples
            if print_on && i==1
                fprintf('Using topK Sampled\n')
            end
            [y] = my_ProcLevSample_topk(lev_scoresH,s);
            sH = H(y,:); % take rows of H
            sX = X(:,y); % take cols of X
            if track_e
                e_list(1,i) = k-k*sum(lev_scoresH(y));
            end
        else
            % weighted sampling only
            [C,D] = my_ProcLevSample(lev_scoresH,s);
            sH = D*H(C,:);
            sX = X(:,C)*D;
        end 
        output.sampler_time  = output.sampler_time + toc(sampler_start);

        % Update W
        mm_start = tic;
        HtXt = sH'*sX';
        HtH = sH'*sH;
        output.matmul_time = output.matmul_time + toc(mm_start);
        solver_start = tic;
        W = NMF_Update(HtH,HtXt,W,'F2',H,'alpha',alpha,'update_rule',alg_str,'doSym',true);
        output.solver_time = output.solver_time + toc(solver_start);


        %% Start update H
        sampler_start = tic;

        % sample from W
        if use_stable_lev % use regular QR or cholesky QR
            [lev_scoresW] = get_Full_LevScores(W);
        else
            [lev_scoresW] = get_Full_LevScores_wCholQR(W);
        end
        % Determine the sampling strategy
        if hybrid_sample
            % partially deterministic sample
            [didx,C,D] = my_ProcLevSample_det(lev_scoresW,s,tau);
            sW = [W(didx,:);D*W(C,:)];
            sX = [Xt(:,didx),Xt(:,C)*D];
            if track_theta
                t_list(2,i) = sum(lev_scoresH(C)); % sum of deterministic lvs scores
                t_list(4,i) = sum(numel(C)); % sum of deterministic lvs scores
            end
        elseif topk_sample
            % topk samples
            y = my_ProcLevSample_topk(lev_scoresW,s);
            sW = W(y,:);
            sX = Xt(:,y);
            if track_e 
                e_list(2,i) = k-k*sum(lev_scoresW(y));
            end
        else
            % weighted sampling only
            [C,D] = my_ProcLevSample(lev_scoresW,s);
            sW = D*W(C,:);
            sX = Xt(:,C)*D;
        end
        output.sampler_time  = output.sampler_time + toc(sampler_start);
        
        % update H
        mm_start = tic;
        WtX = sW'*sX';
        WtW = sW'*sW;
        output.matmul_time = output.matmul_time + toc(mm_start);

        solver_start = tic;
        H = NMF_Update(WtW,WtX,H,'F2',W,'alpha',alpha,'update_rule',alg_str,'doSym',true);
        output.solver_time = output.solver_time + toc(solver_start);

        %% Update the outputs
        output.time(i+1) = toc(loop_start);
        WtX = W'*X;
        cres = efficient_GetRes(normX,X,W,W,'WtX',WtX);
        output.relres(i+1) = cres;
        if prjG
            output.prj_grad(i+1) = efficient_GetPrjGradNrm(X,W,W,'WtX',WtX,'tol',eps,'doSym',true);
        end
        if print_on
            fprintf('Iter: %d, Relative residual : %f\n',i, output.relres(i+1));
        end
        
        if i > min_iters
            if (sum(output.time) >= maxtime) 
                break;
            end
        end
    end % main iteration loop
    
    cresW = efficient_GetRes(normX,X,W,W);
    cresH = efficient_GetRes(normX,X,H,H);
    if cresH < cresW % return the factor with the better approximation...
        W = H;
    end
    output.e_list = e_list;
    if print_on
        sprintf('FINAL, Relative residual : %f', min(cresW,cresH));
    end
    output.params = inParams.Results;
end