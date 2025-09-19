function [Q,S] = symmetric_QB(X,l,q,dist_str,postprocess)
%% Get a structured Sketch Matrix
% This function computes a structured Sketch matrix.
% This is mostly used in the randomized dense NMF routines.
% In this method we assume that X is symmetric
% If postprocess is true then the factor will W = Q*U
    [~,n] = size(X);
    if ~exist('dist_str','var')
        dist_str = 'gaussian';
    end
    if ~exist('postprocess','var')
        postprocess = false; 
    end
    
    if strcmp(dist_str,'gaussian')
        Omega = normrnd(0,1/l,[n,l]);
    elseif strcmp(dist_str,'uniform')
        Omega = rand([n,l]);
    else
        error('getSS() recieved an invalid dist_str.[gaussian,uniform]\nRecieved : %s\n',dist_str);
    end
    
    Y = X*Omega;
    [Q,~]   = qr(Y,0);
    for j = 1:q
        [Q,~] = qr(X'*Q,0);
    end
    
    T = (Q'*X)*Q;
    if postprocess
        T = (T+T')/2; % symmetrize T
        [U,S] = eig(T);
        Q = Q*U;
    else 
        S = T;
    end
end
