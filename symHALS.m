function [U,V,output,acch] = symHALS(M,r,varargin)

[m,~] = size(M);
U = 2 * full(sqrt(mean(mean(M)) / r)) * rand(m, r);
V = 2 * full(sqrt(mean(mean(M)) / r)) * rand(r, m);
maxiter = 1000;
truelabel = [];
lambda = 1;

if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'MAX_ITER',    maxiter=varargin{i+1};
            case 'U_INIT',      U=varargin{i+1};
            case 'V_INIT',      U=varargin{i+1};
            case 'LAMBDA',      lambda=varargin{i+1};
            case 'TRUELABEL',   truelabel=varargin{i+1};
            otherwise
                error(['Unrecognized option: ',varargin{i}]);
        end
    end
end


alpha = 0;
delta = 0;
if size(V,1) > size(V,2)
    V = V';
end

% Initialization
output.relres = [];
output.diff = [];
output.time = [];
output.relres_time = [];

normX = norm(M,'fro');
etime = tic; 
[m,n] = size(M); [m,r] = size(U);

acch = 0;

% initial error
XH = M*V';
cres = efficient_GetRes(normX,M,U,V','XH',XH);
output.relres(1) = cres;
output.diff(1)   = norm(U-V','fro')^2;
output.time(1)   = 0;

if ~isempty(truelabel)
    [~,labelp] = max(V', [], 2);
    tempaccp = ClusteringMeasure(truelabel, labelp);
    acch(1) = tempaccp(1);
end

% Main loop
iter = 0;
while iter <= maxiter
    loop_tic = tic;
    
    % --- Update U ---
    M1 = [M sqrt(lambda)*V']; 
    V1 = [V sqrt(lambda)*eye(r)];
    A = M1*V1'; 
    B = V1*V1'; 
    U = HALSupdt(U',B',A',0,alpha,delta); U = U';
    
    % --- Update V ---
    M2 = [M; sqrt(lambda)*U']; 
    U2 = [U; sqrt(lambda)*eye(r)];
    A = U2'*M2; 
    B = U2'*U2;
    V = HALSupdt(V,B,A,0,alpha,delta);
    
    % --- Record errors & timing ---
    output.time(iter+2) = toc(loop_tic);
    
    res_time_tic = tic;
    XH = M*V';
    cres = efficient_GetRes(normX,M,U,V','XH',XH);
    output.relres(iter+2) = cres;
    output.diff(iter+2)   = norm(U-V','fro')^2;
    output.relres_time(iter+2) = toc(res_time_tic);
    
    if ~isempty(truelabel)
        [~,labelp] = max(V', [], 2);
        tempaccp = ClusteringMeasure(truelabel, labelp);
        acch(iter+2) = tempaccp(1);
    end
    
    iter = iter + 1;
end

if size(V,1) < size(V,2)
    V = V';
end

output.total_time = cumsum(output.time);

end


% === HALSupdt subroutine ===
function V = HALSupdt(V,UtU,UtM,eit1,alpha,delta)
[r,n] = size(V); 
eit2 = cputime; 
cnt = 1; 
eps = 1; eps0 = 1; eit3 = 0;
while cnt == 1 || (cputime-eit2 < (eit1+eit3)*alpha && eps >= (delta)^2*eps0)
    nodelta = 0; 
    if cnt == 1, eit3 = cputime; end
    for k = 1:r
        deltaV = max((UtM(k,:)-UtU(k,:)*V)/UtU(k,k),-V(k,:));
        V(k,:) = V(k,:) + deltaV;
        nodelta = nodelta + deltaV*deltaV';
        if V(k,:) == 0, V(k,:) = 1e-16*max(V(:)); end
    end
    if cnt == 1
        eps0 = nodelta; 
        eit3 = cputime-eit3; 
    end
    eps = nodelta; 
    cnt = 0; 
end
end
