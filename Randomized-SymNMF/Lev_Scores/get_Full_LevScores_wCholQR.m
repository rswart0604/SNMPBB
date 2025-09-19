function [lev_scores] = get_Full_LevScores_wCholQR(W)
%% This function computes all the leverage scores a matrix
% We us the Cholesky QR algorithm to compute the leverage scores
% this method is faster in practice because it relies on matmul
% but is numerically unstable. 
[~,k]       = size(W);
R           = chol(W'*W);
opts.LT     = true;
Q           = linsolve(R',W',opts); % Q is k by m
lev_scores  = (sum(Q.*Q,1)./k)';
end

