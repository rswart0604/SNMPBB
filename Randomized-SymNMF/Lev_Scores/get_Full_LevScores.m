function [lev_scores] = get_Full_LevScores(W)
%% This function computes all the leverage scores a matrix
% Using the QR decomposition the normalized leverage scores of the matrix
% W are computed and returned. 
    [~,k] = size(W);
    [Q,~] = qr(W,0); % economy QR
    lev_scores = sum(Q.*Q,2);
    lev_scores = lev_scores./k;
end

