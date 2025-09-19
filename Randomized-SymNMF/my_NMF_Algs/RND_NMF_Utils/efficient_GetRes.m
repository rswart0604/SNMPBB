function [res] = efficient_GetRes(normX,X,W,H,varargin)
%   This is a function for computing the objective function for NMF
%   efficiently using the trace expansion.
%  ||X-W*H'||

    inParams = inputParser;

    addParameter(inParams,'WtX',[]);
    addParameter(inParams,'XH',[]);
    addParameter(inParams,'HtH',[]);
    addParameter(inParams,'WtW',[]);

    parse(inParams,varargin{:});

    WtX = inParams.Results.WtX;
    XH = inParams.Results.XH;
    HtH = inParams.Results.HtH;
    WtW = inParams.Results.WtW;
    
    if numel(HtH) == 0
        HtH = H'*H;        
    end
    
    if numel(WtW) == 0
        WtW = W'*W;        
    end
    G = trace(WtW * HtH);
    if numel(WtX) ~=0
        WtXH = WtX*H;
    elseif numel(XH) ~=0
        WtXH = W'*XH;
    else % if WtX is not given compute it
        WtXH = (W'*X)*H;
    end 
    %traceHtXW2 = 2*WtX(:)'*H(:); 
    traceHtXW2 = 2*trace(WtXH);
    res = sqrt((normX^2 - traceHtXW2 + G))/normX;
end