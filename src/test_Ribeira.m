clc;
clear;
close all;

rng(101);

%% Load data: it loads a tensor of size 203x268x33.


load('ref_ribeira1bbb_reg1_resize_203x268x33.mat');
A = out;
tensor_dims = size(A);


%% total_entries = prod(tensor_dims);
total_entries = prod(tensor_dims);

fraction = 0.1;% as training data.


nr = 2*round(fraction * total_entries);
indfull = randperm(total_entries);
ind = indfull(1 : nr/2);
P = false(tensor_dims);
P(ind) = true;

% Hence, we know the nonzero entries in PA:
PA = P.*A;

indtest = indfull(nr/2 +1 : nr);
Ptest = false(tensor_dims);
Ptest(indtest) = true;
PAtest = Ptest.*A;

Ilin= find(PA);
PAvec = PA(:);
entries = PAvec(Ilin);
[I, J, K] = ind2sub(tensor_dims,Ilin);
subs = [I J K];

Itestlin= find(PAtest);
PAtestvec = PAtest(:);
entriestest = PAtestvec(Itestlin);
[Itest, Jtest, Ktest] = ind2sub(tensor_dims,Itestlin);
substest = [Itest Jtest Ktest];


data_train.entries = entries; 
data_train.subs = subs; 
data_train.size = tensor_dims; 
data_train.nentries = length(entries);


data_test.nentries = length(entriestest); % Depends how big data can we handle.
data_test.subs = substest;
data_test.entries = entriestest;
data_test.size = tensor_dims;


% % Mean subtraction.
% trainmean = mean(data_train.entries);
% data_train.entries = data_train.entries - trainmean;
% data_test.entries = data_test.entries - trainmean;





%% Call to our algorithms
rank_dims_latent = [5 5 5];  % Rank that we impose.

% Options are not mandatory
opts.maxiter = 100;
opts.tolgradnorm = 1e-5;
opts.tolrelgradnorm = 1e-5;
opts.maxinner = 10;
opts.method = 'tr'; % 'tr' and 'cg'.

problem.tensorsize = tensor_dims;
problem.tensorrank = rank_dims_latent;
problem.subs = data_train.subs;
problem.Y = data_train.entries;
problem.subs_test = data_test.subs;
problem.Y_test = data_test.entries;

% Initialization
xinit.U1 = randn(tensor_dims(1), rank_dims_latent(1));
xinit.U2 = randn(tensor_dims(2), rank_dims_latent(2));
xinit.U3 = randn(tensor_dims(3), rank_dims_latent(3));
xinit.U1 = xinit.U1/norm(xinit.U1, 'fro');
xinit.U2 = xinit.U2/norm(xinit.U2, 'fro');
xinit.U3 = xinit.U3/norm(xinit.U3, 'fro');

%% TR-Dual Alogorithm
% Setting 1.
lambda1 = 1; tensor_dims(1);
lambda2 = 1; tensor_dims(2);
lambda3 = 1; tensor_dims(3);

% % Setting 2.
% lambda1 = tensor_dims(1);
% lambda2 = tensor_dims(2);
% lambda3 = tensor_dims(3);

problem.lambda1 = lambda1;
problem.lambda2 = lambda2;
problem.lambda3 = lambda3;

C = 1e3;
problem.C = C; % Only required for Dual.

opts.pcgmaxiter = 50; % Only required for Dual.
opts.pcgtol = 1e-5; % Only required for Dual.
opts.dotpcgmaxiter = 15; % Only required for Dual.
opts.dotpcgtol = 1e-5; % Only required for Dual.

[Xsol, infos] = dual_fixedrank_latentnorm_tensor_completion(problem, xinit, opts);


%% TR-Primal Alogorithm
lambda1 = tensor_dims(1); % Good setting for Primal.
lambda2 = tensor_dims(2);
lambda3 = tensor_dims(3);
problem.lambda1 = lambda1;
problem.lambda2 = lambda2;
problem.lambda3 = lambda3;

xinit.z = zeros(length(data_train.entries),1); % Primal requires this.

[Xsol_primal, infos_primal] = primal_fixedrank_latentnorm_tensor_completion(problem, xinit, opts);



%% Plots
if ~isempty(data_test)
    % Testing
    fs = 20;
    figure;
    semilogy([infos.time],[infos.loss_test_rmse], '.','Color','red','linewidth', 2.0);
    hold on;
    semilogy([infos_primal.time],[infos_primal.loss_test_rmse], '--','Color','red','linewidth', 2.0);
    hold off;
    ax1 = gca;
    set(ax1,'FontSize',fs);
    xlabel(ax1,'Time in seconds','FontSize',fs);
    ylabel(ax1,'Test RMSE','FontSize',fs);
    legend('TR-Dual','TR-Primal');
    legend 'boxoff';
    box off;
    title(['Fraction known ',num2str(fraction),', test error on a set  \Gamma']) 
end


