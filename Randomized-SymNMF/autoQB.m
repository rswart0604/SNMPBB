function [Q,hist] = autoQB(X,l,varargin)
%% Auto RRF
% This function efficiently computes a QB decomposition and checks the
% residual and or projected gradietn at each iteration.
% The products used to check the gradient or residual can be reused in the
% next power iteration of the QB if desired.
% No symmetry is enforced in this routein.

    inParams = inputParser;

    addParameter(inParams,'isSym',false);
    addParameter(inParams,'Qstart',1);
    addParameter(inParams,'normX',-1);
    addParameter(inParams,'Qmax',8);
    addParameter(inParams,'compPG',false);
    addParameter(inParams,'dist_str','gaussian');
    addParameter(inParams,'tol',1e-3);
    addParameter(inParams,'printon',false);


    parse(inParams,varargin{:});

    isSym       = inParams.Results.isSym;
    q           = inParams.Results.Qstart;
    normX       = inParams.Results.normX;
    Qmax        = inParams.Results.Qmax;
    compPG      = inParams.Results.compPG;
    dist_str    = inParams.Results.dist_str;
    tol         = inParams.Results.tol;
    printon     = inParams.Results.printon;

    resvec = zeros(Qmax,1);
    pgvec  = zeros(Qmax,1);
    
    if normX < 0
        normX = norm(X,'fro');
    end
    
    [~,n] = size(X);
    if strcmp(dist_str,'gaussian')
        Omega = normrnd(0,1/l,[n,l]);
    elseif strcmp(dist_str,'uniform')
        Omega = rand([n,l]);
    else
        error('getSS() recieved an invalid dist_str.[gaussian,uniform]\nRecieved : %s\n',dist_str);
    end
    
    Y = X*Omega;
    [Q,~] = qr(Y,0);
    for j = 1:Qmax % power iteration look
        if isSym
            Y = X*Q; % Bt
            resvec(j) = sqrt(normX^2 - norm(Y,'fro')^2);
            %resvec(j) = efficient_GetRes(normX,X,Q,Y,'XH',Y);
            if compPG
                pgvec(j) = efficient_GetPrjGradNrm(X,Q,Y,'WtX',Y','tol',eps,'doSym',false,'WtW',eye(l));
            end
        else
            Bt = X'*Q;
            resvec(j) = sqrt(normX^2 - norm(Bt,'fro')^2);
            if compPG
                pgvec(j) = efficient_GetPrjGradNrm(X,Q,Bt,'WtX',Bt','tol',eps,'doSym',false,...
                    'WtW',eye(l));
            end
            [Q,~] = qr(Bt,0);
            Y    = X*Q;
        end
        % Reorthogonalize
        [Q,~]   = qr(Y,0);
        %% CHECK RES HERE
        if j > 1
            dfvec = abs(diff(resvec(1:j)./normX));
            if dfvec(end) <= tol
                break;
            end
        end
    end
    if printon
        fprintf('AUTOAB :: q = %d\n',j)
    end
    hist.resvec = resvec;
    hist.pgvec  = pgvec;
end