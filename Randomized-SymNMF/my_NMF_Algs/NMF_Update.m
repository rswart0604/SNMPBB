function [F] = NMF_Update(G,J,F,varargin)
%% This is a function for NMF udpates based on the Alternating Upating (AU)
% framework.
% This supports AU the following rules
% 1) BPP
% 2) HALS
% 3) Least Squares (no nonnegativity constraints)
% Inputs : 
% 1) G, the gram matrix is k by k
% 2) J, should be k by m following the normal equations
% 3) F, the factor matrix being updated, is m by k
% Parameters : ('parameter_string' : 'opt1', 'opt2', ...)
% 1) 'update_rule' : ('bpp', 'hals'), no options is least squares solve
% 2) 'normF' : ('false', 'true'), if to normlize the factor matrix
% 3) 'e' : (positive real number) small pertrubation to add to factors to
% avoid divide by zeros
% 4) 'doSym' : ('false', 'true'), toggle for symmetric regularization
% 5) 'alpha' : (nonnegative real number), regularization parameter
% 6) 'F2' : second factor matrix needed for symNMF update
inParams = inputParser;
addParameter(inParams,'update_rule','bpp');
addParameter(inParams,'normF',false);
addParameter(inParams,'e',1e-16);
addParameter(inParams,'doSym',false);
addParameter(inParams,'alpha',0);
addParameter(inParams,'F2',[]);


parse(inParams,varargin{:});
update_rule     = inParams.Results.update_rule;
normF           = inParams.Results.normF;
e               = inParams.Results.e;
doSym           = inParams.Results.doSym;
alpha           = inParams.Results.alpha;
F2              = inParams.Results.F2;

[~,k] = size(F);

if doSym
    if ~prod(size(F) == size(F2))
        error('In NMF_Update with doSym = True, F and F2 must have the same size.')
    end
    aI_k = alpha*eye(k);
end

if strcmp(update_rule,'hals') % HALS udpate
    for j = 1:k
        if doSym % (HtH)jj*Wj + (XH)j^t - W*HtHj
            % (J(j,:)' - F*G(:,j) + alpha*F2(:,j))/(G(j,j) + alpha) +
            % (G(j,j)*F(:,j))/(G(,j,) + alpha)
            F(:,j) = max((G(j,j)*F(:,j) + J(j,:)' - F*G(:,j) + alpha*F2(:,j))./(G(j,j) + alpha), e); 
        else
            F(:,j) = max(F(:,j) + (J(j,:)' - F*G(:,j))./G(j,j), e); % normalize W columns
            if normF && sum(F(:,j)) > 0
               F(:,j) = F(:,j)./norm(F(:,j));
            end
        end
    end
elseif strcmp(update_rule,'bpp') % BPP update
    if doSym
        F = nnlsm_blockpivot(G + aI_k, (J + alpha * F2'), 1, F')';
        %F = F';
    else
        F = nnlsm_blockpivot(G,J,1,F')';
        %F = F';
    end
else % unconstrained LS
    F = mldivide(G,J)';
    %F = F';
end


end

