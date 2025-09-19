function [W,H,output]=LAI_SNMPBB(V,r,varargin)

% Reference: Yakui Huang, Hongwei Liu, and Shuisheng Zhou: "Quadratic regularization projected Barzilai--Borwein method 
% for nonnegative matrix factorization", Data Mining and Knowledge Discovery, 2015, 29(6): 1665-1684.
% 
% as well as Li, W.; Shi, X. A Gradient-Based Algorithm with Nonmonotone Line Search for Nonnegative Matrix Factorization. 
% Symmetry 2024, 16, 154. https://doi.org/10.3390/sym16020154
%
% Written by Yakui Huang and modified by Ryan Swart


% <Inputs>
%        V : Input data matrix (m x n)
%        r : Target low-rank
%
%        (Below are optional arguments: can be set by providing name-value pairs)
%        MAX_ITER : Maximum number of iterations. Default is 1,000.
%        MIN_ITER : Minimum number of iterations. Default is 1.
%        MAX_TIME : Maximum amount of time in seconds. Default is 100,000.
%        W_INIT : (m x r) initial value for W.
%        H_INIT : (r x n) initial value for H.
%        TOL : Stopping tolerance. Default is 1e-7. If you want to obtain a more accurate solution, decrease TOL and increase MAX_ITER at the same time.
%        VERBOSE : 0 (default) - No debugging information is collected.
%                  1 (debugging purpose) - History of computation is returned by 'HIS' variable.
%                  2 (debugging purpose) - History of computation is additionally printed on screen.
%        SYM_WEIGHT: The weight of the symmetric penalty parameter
% 
% <Outputs>
%        W : Obtained basis matrix (m x r)
%        H : Obtained coefficients matrix (r x n)
%        output : (debugging purpose) History of computation
%
% <Usage Examples>
%        >>V=rand(100);
%        >>LAI_SNMPBB(V,10)
%        >>LAI_SNMPBB(V,20,'verbose',1)
%        >>LAI_SNMPBB(V,30,'verbose',2,'w_init',rand(m,r))
%        >>LAI_SNMPBB(V,5,'verbose',2,'tol',1e-5)



if ~exist('V','var'),    error('please input the sample matrix.\n');    end
if ~exist('r','var'),    error('please input the low rank.\n'); end

[m,n]=size(V);

% Default setting
MaxIter=3000;
MinIter=10;
MaxTime=100000;
W0 = 2 * full(sqrt(mean(mean(V)) / r)) * rand(m, r);
H0= 2 * full(sqrt(mean(mean(V)) / r)) * rand(r,n);
tol=1e-5;
verbose=0;
sym_weight = ((m/10)/(log(r)/log(5)))/10;
p=0;
q=0;

% Read optional parameters
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'MAX_ITER',    MaxIter=varargin{i+1};
            case 'MIN_ITER',    MinIter=varargin{i+1};
            case 'MAX_TIME',    MaxTime=varargin{i+1};
            case 'W_INIT',      W0=varargin{i+1};
            case 'H_INIT',      H0=varargin{i+1};
            case 'TOL',         tol=varargin{i+1};
            case 'VERBOSE',     verbose=varargin{i+1};
            case 'SYM_WEIGHT',  sym_weight=varargin{i+1};
            case 'P',  p=varargin{i+1};
            case 'Q',  q=varargin{i+1};
            case 'DO_AQB',  auto_qb=varargin{i+1};
            otherwise
                error(['Unrecognized option: ',varargin{i}]);
        end
    end
end

ITER_MAX=1000;      % maximum inner iteration number (Default)
ITER_MIN=1;         % minimum inner iteration number (Default)

% Initialization
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

qqq = tic;
if auto_qb
    [U] = autoQB(V,r+p,'isSym',false,'normX',norm(V,'fro'),...
                'isSym',true);
    E = (U'*V)*U;
else
    [U,E] = symmetric_QB(V,r+p,q,'gaussian',true);
end

output.relres = zeros(1,MaxIter+1);
output.time = zeros(1,MaxIter+1);
output.relres_time = zeros(1,MaxIter+1);
output.pgn = zeros(1,MaxIter+1);


% Iterative updating

W=W';
UE = U*E;


first_time = toc(qqq);
normX = norm(V, 'fro');
XH = V*H';
cres = efficient_GetRes(normX,V,H',H','XH',XH);
output.relres(1) = cres;
output.relres_time(1) = toc(qqq);
output.time(1) = first_time;
output.pgn(1) = init_delta;



for iter=1:MaxIter,   
    qqqq = tic;

    % Optimize W with H fixed
    HHt = H * H' + sym_weight * eye(r);
    tmp  = U' * H';
%     tmp2 = E  * tmp;                   % (r+ρ)×m
    approx_HV = UE * tmp;
    HVt = (approx_HV + sym_weight * H')';
    [W,iterW,GradW] = SNMPBB(W,HHt,HVt,ITER_MAX,ITER_MIN,tolW);
    if iterW<=ITER_MIN
        tolW=tolW/10;
    end
    
    
    % Optimize H with W fixed
    WtW = W * W' + sym_weight * eye(r);
    tmp  = U' * W';                    
    tmp2 = E  * tmp;                   
    approx_WV = U * tmp2;              
    WtV = ( approx_WV + sym_weight * W' )';
    [H,iterH,GradH] = SNMPBB(H,WtW,WtV,ITER_MAX,ITER_MIN,tolH);
    if iterH<=ITER_MIN
        tolH=tolH/10;
    end

    delta = norm([GradW(GradW<0 | W>0); GradH(GradH<0 | H>0)]);
    

    output.time(iter+1) = toc(qqqq);
    if verbose
        res_time_start = tic;
        XH = V*H';
        cres = efficient_GetRes(normX,V,H',H','XH',XH);
        output.pgn(iter+1) = delta;
        output.relres(iter+1) = cres;
        output.relres_time(iter+1) = toc(res_time_start);
    end
    
    % Stopping condition
    if iter >= 4
        tcheck = diff(output.relres(iter-3:iter+1))*(-1);
        if prod(tcheck < tol)
            break
        end
    end
    if (delta<=tol*init_delta && iter >= MinIter) || output.time(end)>=MaxTime,
        disp("breaking!!!");
        % break;
    end 
    if delta <= 200*init_delta
        sym_weight = 2*sym_weight;
    end
end
W=W';
output.total_time = cumsum(output.time);
end



function [x,iter,gradx] = SNMPBB(x0,WtW,WtV,iterMax,iter_Min,tol)

s = 1.7;    % relaxation factor
eta = 0.75;

mm=5;       %mm nonmonotone line search parameter
lamax=10^20; lamin=10^-20;
gamma=10^-4;
rho = 0.25; % step length factor

x = x0;     % Initialization
delta0=-sum(sum(x.*WtV)); % HHt, HVt
dQd0 = sum(sum((WtW*x).*x));
f0=delta0+0.5*dQd0;
fn = f0;
lambda=1;

L = 1/norm(full(WtW));    % Lipschitz constant
gradx = WtW*x - WtV;      % Gradient
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
    delta = dx(:)'*gradx(:); % <Zt - Wt, gradx>
    dQd = dx(:)'*dgradx(:); % <Zt - Wt, dgradx>
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
    while (fn > S(iter) + alpha*gamma*(delta))
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

if iter==iterMax,
    fprintf('Max iter in SNMPBB\n');
end

end
