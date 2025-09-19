function [C,D] = my_ProcLevSample(lev_scores,s)
%% This is a function for sampling and formatting scores returned from the
% leverage scores samples.
% In this one we do directed weighted samling based on the given
% leverage scores.
%% Inputs
% 1) lev_scores : any array of normalized leverage scores
% 2) s : the number of desired samples
%% Return
% 1) C the indicies of the rows to be sampled
% 2) D the diagonal s by s reweighting matrix
    n = numel(lev_scores);
    y = randsample(n,s,true,lev_scores);    % draw a random sample based on the weights in lev_scores
    [C,~,ic] = unique(y);                   % get unique values
    counts = histcounts(ic,1:numel(C)+1);   % calculate how many times each value was sampled
    counts = counts';
    D = spdiags((counts).^(1/2).*(lev_scores(C)*s).^(-1/2),0,numel(C),numel(C)); % form diagonal reweighting matrix
end

