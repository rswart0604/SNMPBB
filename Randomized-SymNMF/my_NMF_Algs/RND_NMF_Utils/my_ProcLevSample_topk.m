function [y] = my_ProcLevSample_topk(lev_scores,s,inc_idx,exc_idx)
%% This is a function for sampling and formatting scores returned from the
% leverage scores samples.
% This method retunrs the s indices corresponding to the s highest leverage
% scores in lev_scores.
% This function allows you to pass exc_idx which is a list of indices that
% will be excluded from the topk sampling but included in the output.
%% Return
% 1) y : the indices of the sampled leverage scores

    if exist('exc_idx','var') % set certain lev scores to 0 so they cannot be sampled and renormalized
        lev_scores(exc_idx) = 0;
        lev_scores = lev_scores./sum(lev_scores);
    end

    [~,y] = maxk(lev_scores,s);             % get the top s leverage_scored rows
    if exist('inc_idx','var')               % if given include inc_idx
        y = union(y,inc_idx);
    end
    
end

