function [W,H,output,acc] = NMF_SNMPBB(V,r,varargin)
% NMF_SNMPBB  Symmetric NMF with quadratic regularization using SNMPBB inner solver.
% Refactored to return [W, H, output, acc] like Graph_SNMPBB.
%
% Usage:
%   [W,H,output,acc] = NMF_SNMPBB(V,r,'TRUELABEL',labels,'VERBOSE',2,...)
%
% output fields:
%   output.relres      - relative residual history (||V-W*H|| / ||V||)
%   output.time        - per-iteration elapsed times
%   output.relres_time - time spent computing residual and acc
%   output.total_time  - cumulative times
%
% acc : clustering accuracy history if 'TRUELABEL' provided, otherwise 0

if ~exist('V','var'), error('please input the sample matrix.'); end
if ~exist('r','var'), error('please input the low rank.'); end

[m,n] = size(V);
start_tic = tic;

% ---------------- Default parameters ----------------
MaxIter=750;
MinIter=10;
MaxTime=100000;
W0 = 2 * full(sqrt(mean(mean(V)) / r)) * rand(m, r);
H0 = 2 * full(sqrt(mean(mean(V)) / r)) * rand(r,n);
tol=1e-6;
verbose=0;
sym_weight = (m/10)/(log(r)/log(5)); % weight of symmetric penalty
truelabel = [];

% Parse optional args
for i = 1:2:length(varargin)
    if i+1 > length(varargin), error('Optional parameters should always go by pairs'); end
    name = upper(varargin{i});
    val  = varargin{i+1};
    switch name
        case 'MAX_ITER',   MaxIter = val;
        case 'MIN_ITER',   MinIter = val;
        case 'MAX_TIME',   MaxTime = val;
        case 'W_INIT',     W0 = val;
        case 'H_INIT',     H0 = val;
        case 'TOL',        tol = val;
        case 'VERBOSE',    verbose = val;
        case 'SYM_WEIGHT', sym_weight = val;
        case 'TRUELABEL',  truelabel = val;
        otherwise
            error(['Unrecognized option: ',varargin{i}]);
    end
end

ITER_MAX=1000; ITER_MIN=1;

% ---------------- Initialization ----------------
W=W0; H=H0;
HHt = H * H' + sym_weight * eye(r);
HVt = (V * H' + sym_weight * H')';
WtW = W' * W + sym_weight * eye(r);
WtV = (V * W + sym_weight * W)';
GradW = HHt*W' - HVt;
GradH= WtW*H-WtV;

init_delta = norm([GradW; GradH],'fro');
tolH=max(tol,1e-3)*init_delta;
tolW=tolH;

% ---------------- Histories ----------------
relres_hist = []; t_hist = []; relres_time_hist = [];
etime = tic;

% Initial records
t_hist(1) = toc(start_tic);
relres_tic = tic;
normX = norm(V,'fro');
XH = V*H';
cres = efficient_GetRes(normX,V,H',H','XH',XH);
relres_hist(1) = cres;
relres_time_hist(1) = toc(relres_tic);

if ~isempty(truelabel)
    [~,label0] = max(H,[],1);
    tempacc = ClusteringMeasure(truelabel,label0');
    acc = tempacc(1);
else
    acc = 0;
end

% Transpose W for solver
W = W';

% ---------------- Main loop ----------------
for iter=1:MaxIter
    iter_tic = tic;

    % --- Optimize W ---
    HHt = H * H' + sym_weight * eye(r);
    HVt = (V * H' + sym_weight * H')';
    [W,iterW,GradW] = SNMPBB(W,HHt,HVt,ITER_MAX,ITER_MIN,tolW,V);
    if iterW<=ITER_MIN, tolW=tolW/10; end

    % --- Optimize H ---
    WtW = W * W' + sym_weight * eye(r);
    WtV = (V * W' + sym_weight * W')';
    [H,iterH,GradH] = SNMPBB(H,WtW,WtV,ITER_MAX,ITER_MIN,tolH,V);
    if iterH<=ITER_MIN, tolH=tolH/10; end

    % --- Stopping measure ---
    delta = norm([GradW(GradW<0 | W>0); GradH(GradH<0 | H>0)]);
    
    % --- Time tracking ---
    t_hist(end+1) = toc(iter_tic);
    measure_tic = tic;

    % residual
    XH = V*H';
    cres = efficient_GetRes(normX,V,H',H','XH',XH);
    relres_hist(end+1) = cres;
    
    % clustering accuracy
    if ~isempty(truelabel)
        [~,label] = max(H,[],1);
        tempacc = ClusteringMeasure(truelabel,label');
        acc(end+1) = tempacc(1);
        if tempacc(1) == 1
            break
        end
    end
    relres_time_hist(end+1) = toc(measure_tic);

    
    if (delta<=tol*init_delta && iter>=MinIter) || sum(t_hist)>=MaxTime
        break;
    end
end

% Final transpose
W = W';
elapse = toc(etime);

% ---------------- Pack output struct ----------------
output.relres = relres_hist;
output.time = t_hist;
output.relres_time = relres_time_hist;
output.total_time = cumsum(output.time);

if isempty(acc), acc=0; end

end % NMF_SNMPBB

function [x,iter,gradx] = SNMPBB(x0,WtW,WtV,iterMax,iter_Min,tol,V)
% Quadratic regularization projected Barzilai--Borwein method for the Nonnegative 
% Least Squares Problem: min 1/2 * \|V-Wx\|_{F}^2 subject to x>=0, which is
% equivalent to min 1/2 * <x,WtWx> - <x,WtV> subject to x>=0.

% NEW
s = 2.1;    % relaxation factor
eta = 0.75;


mm=5;       %mm nonmonotone line search parameter
lamax=10^20; lamin=10^-20;
gamma=10^-4;
rho = 0.25; % step length factor

x = x0;     % Initialization
delta0=-sum(sum(x.*WtV)); % HHt, HVt
WtWx = WtW*x;
dQd0 = sum(sum((WtWx).*x));
f0=delta0+0.5*dQd0;
fn = f0;
lambda=1;

L = 1/norm(full(WtW));    % Lipschitz constant
gradx = WtWx - WtV;      % Gradient
for iter=1:iterMax,

    % Stopping criteria
    if iter>=iter_Min,
        pgn = norm(gradx(gradx < 0 | x > 0));
        if pgn<=tol,
            break;
        end
    end
 
    % caculate a point by using the Lipschitz constant
    dx = max(x - L.*gradx, 0)-x;  % get Zt - Wt
    dgradx = WtW*dx;
    delta = dx(:)'*gradx(:);
    dQd = dx(:)'*dgradx(:);
    x = x + dx;  % x is now Zt (we did Wt + (Zt - Wt))
    fn = fn + delta + 0.5*dQd;
    gradx = gradx + dgradx;
    
    % run a projected Barzilai--Borwein step
    func(iter) = fn;
    if iter==1
        S(1) = fn;
    else
        S(iter) = fn + eta*(S(iter-1)-fn);
    end
    
    dx = max(x - lambda.*gradx, 0)-x;
    dgradx = WtW*dx;
    delta = dx(:)'*gradx(:);
    dQd = dx(:)'*dgradx(:);
    fn = func(iter) + delta + 0.5*dQd;
    alpha=1;
    while (fn > S(iter) + alpha*gamma*(delta)) || alpha > 0
        % Use Backtracking Line Search
        alpha=rho*alpha;
        fn = func(iter)+alpha*delta+0.5*alpha^2*dQd;
    end
    x = x+s.*alpha.*dx;  % say Wt = Zt + alpha*dx
        
    % Compute the BB steplength 
    sty = dQd;
    gradx = gradx + alpha.*dgradx;
    if sty > 0
        sts = dx(:)'*dx(:);
        lambda=min(lamax,max(lamin,sts/sty));
    else
        lambda=lamax;
    end   

    if iter == 1
        eta_old = eta;
        eta = eta/2;
    else
        tmp = (eta+eta_old)/2;
        eta_old = eta;
        eta = tmp;
    end
end
end

