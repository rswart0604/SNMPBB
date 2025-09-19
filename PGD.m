function [U,output,acc] = PGD(X,r,varargin)

[m,~] = size(X);
U = 2 * full(sqrt(mean(mean(X)) / r)) * rand(m, r);
maxiter = 300;
truelabel = [];

if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'MAX_ITER',    maxiter=varargin{i+1};
            case 'U_INIT',      U=varargin{i+1};
            case 'TRUELABEL',   truelabel=varargin{i+1};
            otherwise
                error(['Unrecognized option: ',varargin{i}]);
        end
    end
end


% Initialization
output.relres = [];
output.time = [];
output.pgn = [];
output.relres_time = [];


%initial error
relres_tic = tic;
normX = norm(X, 'fro');
XH = X*U;
cres = efficient_GetRes(normX,X,U,U,'XH',XH);
output.relres(1) = cres;
output.relres_time = toc(relres_tic);
output.time(1) = 0;


if ~isempty(truelabel)
    [~,label] = max(U, [], 2);
    tempacc = ClusteringMeasure(truelabel, label);
    acc(1) = tempacc(1);
end

% Main loop
iter = 0;
while iter <= maxiter 
    qqqq = tic;
    
    UU = U'*U;
    XU = X*U;
    UUt = U*U';
    grad = U*UU - XU;
    step = (1 / (norm(X - UUt, 'fro') + 2*norm(UU, 'fro')))*6;
    U = max(U - step * grad, 0);
    
    output.time(iter+2) = toc(qqqq);
    
    res_time_start = tic;
    XH = X*U;
    cres = efficient_GetRes(normX,X,U,U,'XH',XH);
    output.relres(iter+2) = cres;
    if ~isempty(truelabel)
        [~,label] = max(U, [], 2);
        tempacc = ClusteringMeasure(truelabel, label);
        acc(iter+2) = tempacc(1);
    end
    output.relres_time(iter+2) = toc(res_time_start);

%     if iter > 1
%         if (output.relres(iter+2)-output.relres(iter+1))/output.relres(iter+1) < 1e-16
%             break
%         end
%     end

    iter = iter + 1;
end

if nargin < 4
    acc = 0;
end

output.total_time = cumsum(output.time);

