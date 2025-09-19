function [didx,C,D] = my_ProcLevSample_det(lev_scores,s,tau,inc_idx,exc_idx)
%% This is a function for sampling and formatting scores returned from the
% leverage scores samples.
% In this method scores above a threshold (tau) are all taken while the
% rest are sampled randomly.
% This function also allows one to give a list of indices that must be
% included in the sample.
%% Return
% 1) didx the sample with leverage score greater than tau
% 2) C the indicies of the rows to be sampled
% 3) D the diagonal s by s reweighting matrix
% 4) inc_idx, optional list of indicies to always include
    n = numel(lev_scores);
    
    if exist('exc_idx','var')
        lev_scores(exc_idx) = 0;
        lev_scores = lev_scores./sum(lev_scores);
    end
    
    % find the scores greater than tau
    didx = lev_scores > tau;                     % scores to take
    if exist('inc_idx','var')                   % if given include inc_idx
        didx = union(didx,inc_idx);
    end
    sh_rnd = max(s - nnz(didx),0);
    lev_scores(didx) = 0;                        % remove the deterministically sampled leverage score
    lev_scores = lev_scores/sum(lev_scores);     % renormalize

    y = randsample(n,sh_rnd,true,lev_scores);   % draw a random sample from remaining lev scores
    [C,~,ic] = unique(y);                       % get unique values
    counts = histcounts(ic,1:numel(C)+1);       % calculate how many times each value was sampled
    counts = counts';
    D = spdiags((counts).^(1/2).*(lev_scores(C)*s).^(-1/2),0,numel(C),numel(C));
end

