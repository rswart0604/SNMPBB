function [pg] = efficient_GetPrjGradNrm(X,W,H,varargin)
%   This is a function for computing the objective function for NMF
%   efficiently using the trace expansion.
%  ||X-W*H'||
%  W*(HtH) - X*H
% or 

    inParams = inputParser;

    addParameter(inParams,'XH',[]); % X*H or X'*W
    addParameter(inParams,'WtX',[]); % X*H or X'*W
    addParameter(inParams,'WtW',[]); % WtW or HtH
    addParameter(inParams,'HtH',[]); % WtW or HtH
    addParameter(inParams,'tol',eps); % WtW or HtH
    addParameter(inParams,'doSym',false); % WtW or HtH
    
    parse(inParams,varargin{:});

    XH = inParams.Results.XH;
    WtX = inParams.Results.WtX;
    WtW = inParams.Results.WtW;
    HtH = inParams.Results.HtH;
    tol = inParams.Results.tol;
    doSym = inParams.Results.doSym;
    
    
    if doSym
        W = H;
        if numel(HtH) ~= 0
            HtH = HtH;
        elseif numel(WtW) ~= 0 
            HtH = WtW;
        else
            HtH = H'*H;
        end
        
        if numel(XH) ~= 0 
            XH = XH;
        elseif numel(WtX) ~= 0 
            XH = WtX';
        else
            XH = X*H;
        end
        
        FH = 4*(H*HtH - XH);
        pgH = norm(((H>tol) | (FH < 0)).*FH,'fro')^2;
        pg = sqrt(pgH);
    else
        if numel(HtH) == 0
            HtH = H'*H;        
        end
        if numel(XH) == 0
            XH = X*H;
        end   
        if numel(WtW) == 0
            WtW = W'*W;        
        end
        if numel(WtX) == 0
            WtX = W'*X;
        end

        FW = 2*(W*HtH - XH);
        FH = 2*(H*WtW - WtX');
        pgW = norm(((W>tol) | (FW < 0)).*FW,'fro')^2;
        pgH = norm(((H>tol) | (FH < 0)).*FH,'fro')^2;
        pg  = sqrt(pgW + pgH);
    end

end
