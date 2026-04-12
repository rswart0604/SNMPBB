
true_labels = gnd(:,2);
pred_labels = node_clusters;
% Compute metrics
ari_val = adjustedRandIndex(true_labels, pred_labels);
nmi_val = normalizedMutualInfo(true_labels, pred_labels);

fprintf('Adjusted Rand Index (ARI): %.4f\n', ari_val);
fprintf('Normalized Mutual Information (NMI): %.4f\n', nmi_val);


function ari = adjustedRandIndex(labels_true, labels_pred)
    % Ensure both inputs are column vectors
    labels_true = labels_true(:);
    labels_pred = labels_pred(:);

    n = length(labels_true);
    if length(labels_pred) ~= n
        error('Label vectors must be the same length');
    end

    % Contingency table
    [~, ~, idx_true] = unique(labels_true);
    [~, ~, idx_pred] = unique(labels_pred);
    contingency = accumarray([idx_true idx_pred], 1);

    nij = contingency;
    ni = sum(nij, 2);
    nj = sum(nij, 1);

    sum_nij2 = sum(nij(:) .* (nij(:) - 1)) / 2;
    sum_ni2 = sum(ni .* (ni - 1)) / 2;
    sum_nj2 = sum(nj .* (nj - 1)) / 2;

    expected_index = sum_ni2 * sum_nj2 / (n * (n - 1) / 2);
    max_index = (sum_ni2 + sum_nj2) / 2;
    index = sum_nij2;

    ari = (index - expected_index) / (max_index - expected_index);
end
function nmi_val = normalizedMutualInfo(labels_true, labels_pred)
    labels_true = labels_true(:);
    labels_pred = labels_pred(:);

    n = length(labels_true);
    if length(labels_pred) ~= n
        error('Label vectors must be the same length');
    end

    % Contingency table
    [~, ~, idx_true] = unique(labels_true);
    [~, ~, idx_pred] = unique(labels_pred);
    contingency = accumarray([idx_true idx_pred], 1);

    nij = contingency;
    ni = sum(nij, 2);
    nj = sum(nij, 1);
    N = sum(nij(:));

    % Mutual information
    MI = 0;
    for i = 1:size(nij,1)
        for j = 1:size(nij,2)
            if nij(i,j) > 0
                MI = MI + (nij(i,j)/N) * log((nij(i,j)*N) / (ni(i)*nj(j)));
            end
        end
    end

    % Entropies
    H_true = -sum((ni/N) .* log(ni/N + eps));
    H_pred = -sum((nj/N) .* log(nj/N + eps));

    nmi_val = MI / sqrt(H_true * H_pred);
end
