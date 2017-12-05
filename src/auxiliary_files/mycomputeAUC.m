function[auc_roc] = mycomputeAUC(predictions, testdata)
% AUC Computation following the procedure described in [1].
% [1] Lu,  L.,  and  Zhou,  T.   2011.   
% "Link  prediction  in  complex networks: A survey", 
% Physica A: Statistical Mechanics andits Applications, 390(6):1150–1170.


    n = length(testdata);
    positive_samples = find(testdata > 0); %% indices of positive samples
    negative_samples = find(testdata <= 0); %% indices of negative samples
    
    m = 10*n;% 10*n; % Number of pairs considered.
%     if m > 1e6
%         m = 1e6;
%     end
%     m = 1e6;
    
    % Randomly generate indices 
    idx_rand_pos = randi(length(positive_samples), m, 1);
    idx_rand_neg = randi(length(negative_samples), m, 1);
    
    positive_idx = positive_samples(idx_rand_pos);
    negative_idx = negative_samples(idx_rand_neg);
    
    auc_roc = (sum(predictions(positive_idx) > predictions(negative_idx)) + 1/2 * sum(predictions(positive_idx) == predictions(negative_idx)))/m;
    %     auc_roc = round(auc_roc, 2); % Rounding the auc.
    
end
