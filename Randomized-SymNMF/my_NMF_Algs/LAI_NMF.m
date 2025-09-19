function [W,H,output,U] = LAI_NMF(X,k,varargin)
%% LIA-NMF : Low-rank Approximate Input NMF
% This function is different because it allows for the use of the
% NMF_Update Function
% X \approx U*V'
% 1) X the original data matrix
% 2) k is the desired low-rank
% Below are some optional inputs :
% 3) varargin, various other inputs including but not limited too:
% - U is the right factor of the approximation
% - V is the right factor of the approximation
% - p is used to compute the LRA if its not provided

    %% Parse inputs
    inParams = inputParser;
    addParameter(inParams,'max_iters',50);
    addParameter(inParams,'min_iters',10);
    addParameter(inParams,'print_on',false); % print some debugging info
    addParameter(inParams,'init',{}); % optional factor matrix inits
    addParameter(inParams,'normW',false); % to normalize W or not
    addParameter(inParams,'e',eps); % avoid div by zero epsilon for HALS
    addParameter(inParams,'p',0); % column over sampling params
    addParameter(inParams,'q',0); % power iteration param
    addParameter(inParams,'alg_str','bpp'); % update rule to use for nmf
    addParameter(inParams,'alpha',-1); % regularlizer for symnmf
    addParameter(inParams,'tol',1e-5); % stopping tolerance for residual
    addParameter(inParams,'labels',[]); % labels for a clustering problem
    addParameter(inParams,'do_res_iter',false); % toggle for Iterative Refinement
    addParameter(inParams,'maxIR',5); % max IR iterations
    addParameter(inParams,'minIR',4); % min IR iterations
    addParameter(inParams,'maxtime',Inf); % max time in seconds
    addParameter(inParams,'doSym',false); % do SymNMF toggle
    addParameter(inParams,'prjG',false); % computer projected gradient
    addParameter(inParams,'dist_str','gaussian'); % sketch matrix distribution
    addParameter(inParams,'do_AQB',false); % to use auto QB or not

    parse(inParams,varargin{:});
    
    max_iters   = inParams.Results.max_iters;
    min_iters   = inParams.Results.min_iters;
    print_on    = inParams.Results.print_on;
    init        = inParams.Results.init;
    e           = inParams.Results.e;
    p           = inParams.Results.p;
    q           = inParams.Results.q;
    alg_str    = inParams.Results.alg_str;
    alpha       = inParams.Results.alpha;
    tol         = inParams.Results.tol;
    labels      = inParams.Results.labels;
    do_res_iter = inParams.Results.do_res_iter;
    maxIR       = inParams.Results.maxIR;
    minIR       = inParams.Results.minIR;
    maxtime     = inParams.Results.maxtime;
    doSym       = inParams.Results.doSym;
    prjG        = inParams.Results.prjG;
    dist_str    = inParams.Results.dist_str;
    do_AQB      = inParams.Results.do_AQB;
    
    %% Init some algorithm parameters
    [m,n] = size(X);
    output.relres           = zeros(1,max_iters+1);
    output.apx_relres       = zeros(1,max_iters+1);
    output.time             = zeros(1,max_iters+1);
    output.relres_time      = zeros(1,max_iters+1);
    output.apx_relres_time  = zeros(1,max_iters+1);
    output.prj_grad         = zeros(1,max_iters+1);
    
    if print_on
        fprintf('START, Relative residual : %f\n', output.relres(1));
    end
    
    %% Check if a low-rank approximation of X has been provided
    tic; % TIC
    normX = norm(X,'fro');
    
    % initialize the factor matrices
    if doSym
        if do_AQB
            [U] = autoQB(X,k+p,'isSym',false,'normX',normX,...
                'isSym',doSym);
            E = (U'*X)*U;
        else
            [U,E] = symmetric_QB(X,k+p,q,dist_str,true);
        end
        V = U*E;
        normQB = norm(E,'fro');

        if isempty(init)
            H = 2 * full(sqrt(mean(mean(X)) / k)) * rand(n, k);
        else
            H = init.H;
        end
        W = H;
    else % init for non symmetric
        if do_AQB
            [U] = autoQB(X,k+p,'isSym',false,'normX',normX,...
                'isSym',doSym);
        else
            U = getSS(X,k+p,q,dist_str);
        end
        V   = X'*U;
        normQB = norm(V,'fro');

        if isempty(init)
            H = 2 * full(sqrt(mean(mean(X)) / k)) * rand(n, k);
            W = 2 * full(sqrt(mean(mean(X)) / k)) * rand(m, k);
        else
            H = init.H;
            W = init.W;
        end
    end

    % Set alpha
    if alpha < 0 
        alpha = max(max(X))^2; % just compute the diagonal elements
    end
    RIon = false;
    output.time(1)  = toc; 
    
    % compute initial residuals and time them
    tic;
    WtX = W'*X;
    if doSym; cres = efficient_GetRes(normX,X,W,W,'WtX',WtX); else; 
                cres = efficient_GetRes(normX,X,W,H,'WtX',WtX); end;
    output.relres(1) = cres;
    output.relres_time(1) = output.relres_time(1) + toc;
    tic;
    WtUVt = (W'*U)*V';
    if doSym; apx_cres = efficient_GetRes(normQB,X,W,W,'WtX',WtUVt);; else; 
                apx_cres = efficient_GetRes(normQB,X,W,H,'WtX',WtUVt);; end;
    output.apx_relres(1)    = apx_cres;
    output.apx_relres_time(1)  = output.apx_relres_time(1) + toc;
    
    % Compute the Projected Gradient
    if prjG
        output.prj_grad(1) = efficient_GetPrjGradNrm(X,W,H,'WtX',WtX,'tol',eps,'doSym',doSym);
    end
    
    %% Main Iteration
    checkp = 0;
    for i = 1:max_iters
        tic;
        
        %% update W
        if ~RIon
            HtXt = (H'*V)*U'; % V then U
        else
            HtXt = H'*X';
        end
        HtH     = H'*H;
        W   = NMF_Update(HtH,HtXt,W,'update_rule',alg_str,'e',e,'alpha',alpha,'F2',H,'doSym',doSym);
        %% update H
        if ~RIon
            WtX = (W'*U)*V'; 
        else
            WtX = W'*X;
        end
        WtW = W'*W;
        H = NMF_Update(WtW,WtX,H,'update_rule',alg_str,'e',e,'alpha',alpha,'F2',W,'doSym',doSym);
        %% Update the outputs
        output.time(i+1)      = output.time(i+1) + toc;
        tic;
        if RIon
            J  = WtX;
            JH = HtXt';
        else
            J = W'*X; % this is ok to time bc apx_rel_res is the what we plot
            JH = X*H;
        end
        if doSym; cres = efficient_GetRes(normX,X,W,W,'WtX',J); else; 
                cres = efficient_GetRes(normX,X,W,H,'WtX',J); end;
        output.relres(i+1) = cres;
        output.relres_time(i+1) = output.relres_time(i+1) + toc;
        
        if ~RIon
            tic;
            if doSym; cres = efficient_GetRes(normQB,X,W,W,'WtX',WtX); else; 
                cres = efficient_GetRes(normQB,X,W,H,'WtX',WtX); end;
            output.apx_relres(i+1)      = cres;
            output.apx_relres_time(i+1) = output.apx_relres_time(i+1) + toc;
        else
            output.apx_relres(i+1)      = output.relres(i+1);
            output.apx_relres_time(i+1) = output.relres_time(i+1);
        end
        % compute projected gradient
        if prjG
            output.prj_grad(i+1) = efficient_GetPrjGradNrm(X,W,H,'WtW',WtW,'HtH',HtH,'WtX',J,'XH',JH,'tol',eps,'doSym',doSym);
        end
        
        if print_on
            fprintf('Iter: %d, Relative residual : %f\n',i, output.relres(i+1));
        end
        
        %% check stopping conditions
        if i > min_iters
            % check if there is time left
            if (sum(output.time + output.apx_relres_time) >= maxtime) 
                break;
            end
            
            tcheck = diff(output.apx_relres(i-3:i+1))*(-1);
            if RIon
                checkp = checkp + 1; % increase the RI iterations
                if (checkp > minIR) && prod(tcheck < tol)
                    break; % if we are over the min RI's and there is insufficient decrease
                end
                if checkp > maxIR 
                    break;
                end
            end
            
            % check if the residual has decreased sufficiently     
            if ~RIon && prod(tcheck < tol) % only do this if IR is off
               if do_res_iter && ~RIon % one save for IR
                   RIon = true;
                   output.RIstart = i+1;
               else
                   break;
               end    
            end
        end

    end
    
    
    %% Compute final quantities 
    if doSym
        Wres = efficient_GetRes(normX,X,W,W);
        Hres = efficient_GetRes(normX,X,H,H);
        if  Hres < Wres
            W = H;
        else
            H = W;
        end
    end
    
    if print_on
        sprintf('FINAL, Relative residual : %f', output.relres(end));
    end
    
    if numel(labels) > 0
        [~,clabs] = max(W,[],2);
         output.ari = rand_index(labels, clabs, 'adjusted');
    else
        output.ari = NaN;
    end

    output.params = inParams.Results;
end




