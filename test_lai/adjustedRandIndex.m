function ari = adjustedRandIndex(trueLabels, predLabels)
    % Ensure labels are row vectors
    trueLabels = trueLabels(:);
    predLabels = predLabels(:);
    n = length(trueLabels);

    % Contingency table
    [~,~,gt] = unique(trueLabels);
    [~,~,pr] = unique(predLabels);
    cont = accumarray([gt pr], 1);

    nij = cont;
    ai = sum(nij,2);
    bj = sum(nij,1);

    nC2 = nchoosek(n,2);

    sum_nijC2 = sum(sum(nchoosekVec(nij(:),2)));
    sum_aiC2  = sum(nchoosekVec(ai(:),2));
    sum_bjC2  = sum(nchoosekVec(bj(:),2));

    expected = sum_aiC2 * sum_bjC2 / nC2;
    maxIndex = 0.5*(sum_aiC2 + sum_bjC2);

    ari = (sum_nijC2 - expected) / (maxIndex - expected);
end

function out = nchoosekVec(v,k)
    % Vectorized nchoosek(v,2)
    v = v(:);
    out = v.*(v-1)/2;
end
