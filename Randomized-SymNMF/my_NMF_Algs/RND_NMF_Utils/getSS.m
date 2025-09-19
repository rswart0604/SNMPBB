function [Q] = getSS(X,l,q,dist_str,varargin)
%% Get a structured Sketch Matrix
% This function computes a structured Sketch matrix.
% This is mostly used in the randomized dense NMF routines.

    inParams = inputParser;

    addParameter(inParams,'use_stable',true);
    addParameter(inParams,'isSym',false);

    parse(inParams,varargin{:});

    use_stable  = inParams.Results.use_stable;
    isSym       = inParams.Results.isSym;
    
    [~,n] = size(X);
    if strcmp(dist_str,'gaussian')
        Omega = normrnd(0,1/l,[n,l]);
    elseif strcmp(dist_str,'uniform')
        Omega = rand([n,l]);
    else
        error('getSS() recieved an invalid dist_str.[gaussian,uniform]\nRecieved : %s\n',dist_str);
    end
    
    Y = X*Omega;
    if use_stable % use the stable method
        for j = 1:q
            [Q,~] = qr(Y,0);
            [Q,~] = qr(X'*Q,0);
            Y    = X*Q;
        end
    else % use the unstable method
        for j = 1:q
            if isSym
                Y = X*Y;
            else
                Y = X'*Y;
                Y = X*Y;
            end
        end
    end
    [Q,~]   = qr(Y,0);
end