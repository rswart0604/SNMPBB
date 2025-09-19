function [H,output,U] = LAI_SymPGNCG(X,k,varargin)
%% SymNMF via PGNCG, Projected Gauss-Newton with Conjugate Gradients
% This function implements the PGNCG method for SymNMF with LAI.
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
    addParameter(inParams,'print_on',false);
    addParameter(inParams,'init',{});
    addParameter(inParams,'normW',false);
    addParameter(inParams,'e',eps);
    addParameter(inParams,'p',0);
    addParameter(inParams,'q',0);
    addParameter(inParams,'QB_init',false);
    addParameter(inParams,'tol',1e-5);
    addParameter(inParams,'labels',[]);
    addParameter(inParams,'do_res_iter',false);
    addParameter(inParams,'maxIR',5);
    addParameter(inParams,'minIR',4);
    addParameter(inParams,'maxtime',Inf);
    addParameter(inParams,'prjG',false);
    addParameter(inParams,'dist_str','gaussian');
    addParameter(inParams,'maxs',5);
    addParameter(inParams,'do_AQB',false);

    parse(inParams,varargin{:});
    
    max_iters   = inParams.Results.max_iters;
    min_iters   = inParams.Results.min_iters;
    print_on    = inParams.Results.print_on;
    init        = inParams.Results.init;
    e           = inParams.Results.e;
    p           = inParams.Results.p;
    q           = inParams.Results.q;
    QB_init     = inParams.Results.QB_init;
    tol         = inParams.Results.tol;
    labels      = inParams.Results.labels;
    do_res_iter = inParams.Results.do_res_iter;
    maxIR       = inParams.Results.maxIR;
    minIR       = inParams.Results.minIR;
    maxtime     = inParams.Results.maxtime;
    prjG        = inParams.Results.prjG;
    dist_str    = inParams.Results.dist_str;
    maxs        = inParams.Results.maxs;
    do_AQB      = inParams.Results.do_AQB;

    %% Init some algorithm parameters
    [n,~] = size(X);
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
    % init the factors
    if do_AQB
        [U] = autoQB(X,k+p,'isSym',false,'normX',normX,...
                'isSym',true);
        E = (U'*X)*U;
    else
        [U,E] = symmetric_QB(X,k+p,q,dist_str,true);
    end
    V = U*E;
    normQB = norm(E,'fro');
    if QB_init
        [~,idmax] = maxk(sum(E.*E),k);
        H = V(:,idmax);
    elseif isempty(init)
        H = 2 * full(sqrt(mean(mean(X)) / k)) * rand(n, k);
    else
        H = init.H;
    end
    
    RIon = false;
    output.time(1)  = toc; 
    
    % compute initial residuals and time them
    tic;
    XH = X*H;
    cres = efficient_GetRes(normX,X,H,H,'XH',XH);
    output.relres(1) = cres;
    output.relres_time(1) = output.relres_time(1) + toc;
    tic;
    UVtH = U*(V'*H);
    apx_cres = efficient_GetRes(normQB,X,H,H,'XH',UVtH); 
    output.apx_relres(1)    = apx_cres;
    output.apx_relres_time  = output.apx_relres_time + toc;
    % Compute the Projected Gradient
    if prjG
        output.prj_grad(1) = efficient_GetPrjGradNrm(X,H,H,'XH',XH,'tol',eps);
    end
    %% Main Iteration
    checkp = 0;
    for i = 1:max_iters
        tic;
        
        %% update W
        if ~RIon
            XH = U*(V'*H); % V then U
        else
            XH = X*H;
        end
        Z = zeros(size(H));
        % Compute Gradient
        HtH = H'*H;
        R = -2*(XH-H*HtH);
        P = R;
        e_old = norm(R,'fro')^2;
        for sidx = 1:maxs
            Y = Apply_Gramian(H,P,HtH);
            alpha = e_old/sum(sum(P.*Y));
            Z = Z + alpha*P;
            R = R - alpha*Y;
            e_new = norm(R,'fro')^2;
            P = R + (e_new/e_old)*P;
            e_old = e_new;
        end
        H = max(H - Z,0);

        %% Update the outputs
        output.time(i+1)      = output.time(i+1) + toc;
        tic;
        if RIon
            J  = XH;
        else
            J = X*H; % this is ok to time bc apx_rel_res is the what we plot
        end
        cres = efficient_GetRes(normX,X,H,H,'XH',J);
        output.relres(i+1) = cres;
        output.relres_time(i+1) = output.relres_time(i+1) + toc;
        
        if ~RIon
            tic;
            cres = efficient_GetRes(normQB,X,H,H,'XH',XH);
            output.apx_relres(i+1)      = cres;
            output.apx_relres_time(i+1) = output.apx_relres_time(i+1) + toc;
        else
            output.apx_relres(i+1)      = output.relres(i+1);
            output.apx_relres_time(i+1) = output.relres_time(i+1);
        end
        % compute projected gradient
        if prjG
            output.prj_grad(i+1) = efficient_GetPrjGradNrm(X,H,H,'WtW',HtH,'HtH',HtH,'WtX',J','XH',J,'tol',eps);
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
                checkp = checkp + 1; % incrase the RI iterations
                if (checkp > minIR) && prod(tcheck < tol)
                    break; % if we are over the min RI's and there is insufficient decrease
                end
                if checkp > maxIR 
                    break;
                end
            end
            
            % check if the residual has decreased sufficiently     
            if ~RIon && prod(tcheck < tol) % only do this if RI is off
               if do_res_iter && ~RIon % one save for residual iteration
                   %fprintf("Turning on RI\n")
                   RIon = true;
                   output.RIstart = i+1;
               else
                   break;
               end    
            end
        end

    end
    
    
    %% Compute final quantities 
    if print_on
        sprintf('FINAL, Relative residual : %f', output.relres(end));
    end
    
    if numel(labels) > 0
        [~,clabs] = max(H,[],2);
         output.ari = rand_index(labels, clabs, 'adjusted');
    else
        output.ari = NaN;
    end

    output.params = inParams.Results;
end

function [Y] = Apply_Gramian(H,X,HtH)
    V = X'*H;
    Y = X*HtH;
    Y = 2*(Y + H*V);
end





