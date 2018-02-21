function[Xopt, infos, options] = primal_fixedrank_latentnorm_tensor_completion(problem, Uinit, options)
    
    %% Problem Description
    lambda1 = problem.lambda1;
    lambda2 = problem.lambda2;
    lambda3 = problem.lambda3;
    
    
    tensorsize = problem.tensorsize;
    tensorrank = problem.tensorrank;
    
    n1 = tensorsize(1);
    n2 = tensorsize(2);
    n3 = tensorsize(3);
    r1 = tensorrank(1);
    r2 = tensorrank(2);
    r3 = tensorrank(3);
    
    I = problem.subs(:,1); % subscripts corresponding to known entries in tensor along mode 1.
    J = problem.subs(:,2); % subscripts corresponding to known entries in tensor along mode 2.
    K = problem.subs(:,3); % subscripts corresponding to known entries in tensor along mode 3.
    Y = problem.Y; % Known entries.
    length_entries = length(I); % Number of known entries.
    
    
    %% Generate indices for the given data
    
    [tensorindices, converters]= generate_indices(problem.subs, problem.tensorsize);
    
    
    tensortomode1 = converters.tensortomode1;
    tensortomode2 = converters.tensortomode2;
    tensortomode3 = converters.tensortomode3;
    
    mode1totensor = converters.mode1totensor;
    mode2totensor = converters.mode2totensor;
    mode3totensor = converters.mode3totensor;
    
    mode2tomode1 = converters.mode2tomode1;
    mode3tomode1 = converters.mode3tomode1;
    mode1tomode2 = converters.mode1tomode2;
    mode1tomode3 = converters.mode1tomode3;
    
    J1unique = tensorindices.mode1.Junique;
    J1relative = tensorindices.mode1.Jrelative;
    
    J2unique = tensorindices.mode2.Junique;
    J2relative = tensorindices.mode2.Jrelative;
    
    J3unique =  tensorindices.mode3.Junique;
    J3relative = tensorindices.mode3.Jrelative;
    
    mask1relative = tensorindices.mode1.maskrelative;
    mask2relative = tensorindices.mode2.maskrelative;
    mask3relative = tensorindices.mode3.maskrelative;
    
    mask1 = tensorindices.mode1.mask;
    mask2 = tensorindices.mode2.mask;
    mask3 = tensorindices.mode3.mask;
    
    I1 = tensorindices.mode1.I;
    I2 = tensorindices.mode2.I;
    I3 = tensorindices.mode3.I;
    
    J1 = tensorindices.mode1.J;
    J2 = tensorindices.mode2.J;
    J3 = tensorindices.mode3.J;
    
    I2_1 = I2(mode2tomode1);
    J2_1 = J2(mode2tomode1);
    J2_1relative = J2relative(mode2tomode1);
    
    I3_1 = I3(mode3tomode1);
    J3_1 = J3(mode3tomode1);
    J3_1relative = J3relative(mode3tomode1);
    
    %% Known entries reordered along mode 1.
    Y1 = Y(tensortomode1); % Y1 is the array of entries arranged in the mode 1 ordering.
    
    
    %% Test data if given.
    if isfield(problem,'subs_test') && ~isempty(problem.subs_test)
        
        I_test = problem.subs_test(:,1); % subscripts corresponding to test entries in tensor.
        J_test = problem.subs_test(:,2); % subscripts corresponding to test entries in tensor.
        K_test = problem.subs_test(:,3); % subscripts corresponding to test entries in tensor.
        Y_test = problem.Y_test; % Test entries
        length_entries_test = length(I_test);
        
        % Generate indices
        [tensorindices_test, converters_test]= generate_indices(problem.subs_test, problem.tensorsize);
        
        I1_test = tensorindices_test.mode1.I;
        I2_test = tensorindices_test.mode2.I;
        I3_test = tensorindices_test.mode3.I;
        
        J1_test = tensorindices_test.mode1.J;
        J2_test = tensorindices_test.mode2.J;
        J3_test = tensorindices_test.mode3.J;
        
        % Only required if a test set is given.
        V1_full = zeros(r1, size(mask1, 2));
        V2_full = zeros(r2, size(mask2, 2));
        V3_full = zeros(r3, size(mask3, 2));
        
    end
    
    %% Local defaults for options
    
    localdefaults.maxiter = 50; % Max iterations.
    localdefaults.tolgradnorm = 1e-6; % Absolute tolerance on Gradnorm.
    localdefaults.method = 'TR'; % Default solver is trustregions (TR).
    localdefaults.verbosity = 2; % Show output.
    localdefaults.computeauc = false; % Compute the AUC, useful in link prediction applications.
    localdefaults.mu = 0; % No regularization.
    localdefaults.computenmse = false; % Compute NMSE, useful in multitask learning.
    
    
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    
    %% Initialization
    
    if ~exist('Uinit', 'var') % Uinit is a structure.
        Uinit = [];
    end
    
    %% Manifold geometry: the Cartesian product of three spectrahedron manifolds.
    
    % mymanifold.U1 = symfixedrankYYfactory(n1, r1);
    % mymanifold.U2 = symfixedrankYYfactory(n2, r2);
    % mymanifold.U3 = symfixedrankYYfactory(n3, r3);
    
    mymanifold.U1 = spectrahedronfactory(n1, r1);
    mymanifold.U2 = spectrahedronfactory(n2, r2);
    mymanifold.U3 = spectrahedronfactory(n3, r3);
    
    mymanifold.z = euclideanfactory(length(Y1),1);
    %
    %     mymanifold.z = spherefactory(length(Y1),1);
    
    
    problem.M = productmanifold(mymanifold);
    
    %% Cost
    problem.cost = @cost;
    function [f, store] = cost(x, store)
        U1 = x.U1;
        U2 = x.U2;
        U3 = x.U3;
        z = x.z;
        
        
        
        if ~isfield(store, 'Z1')
            
            Z1 = z;
            Z2 = Z1(mode1tomode2);
            Z3 = Z1(mode1tomode3);
            
            
            U1tZ1 = mymultfullsparse(U1', Z1, mask1relative);
            U2tZ2 = mymultfullsparse(U2', Z2, mask2relative);
            U3tZ3 = mymultfullsparse(U3', Z3, mask3relative);
            
            store.U1tZ1 = U1tZ1;
            store.U2tZ2 = U2tZ2;
            store.U3tZ3 = U3tZ3;
            store.Z1 = Z1;
            store.Z2 = Z2;
            store.Z3 = Z3;
            
        end
        U1tZ1 = store.U1tZ1;
        U2tZ2 = store.U2tZ2;
        U3tZ3 = store.U3tZ3;
        
        
        Fac1 = myspmaskmult(U1, U1tZ1, I1, J1relative);
        Fac2_1 = myspmaskmult(U2, U2tZ2, I2_1, J2_1relative);
        Fac3_1 = myspmaskmult(U3, U3tZ3, I3_1, J3_1relative);
        
        
        R = lambda1*Fac1 + lambda2*Fac2_1 + lambda3*Fac3_1 - Y1;
        
        %         % Debug
        %         Fac2 = myspmaskmult(U2, U2tZ2, I2, J2relative);
        %         Fac3 = myspmaskmult(U3, U3tZ3,I3, J3relative);
        %         Rold = Fac1 + Fac2(mode2tomode1) + Fac3(mode3tomode1) - Y1;
        %         norm(R - Rold, 'fro')
        
        
        f = 0.5*norm(R, 'fro')^2  + 0.5*options.mu*(z'*z); % 0.5*(R'*R);
        
        store.R = R; % in mode 1.
        store.R1 = R;
        store.R2 = R(mode1tomode2);
        store.R3 = R(mode1tomode3);
        
        
        %         norm(store.Z3,'fro')
        
    end
    
    
    
    %% Euclidean Gradient
    problem.egrad = @egrad;
    function [grad, store] = egrad(x, store)
        U1 = x.U1;
        U2 = x.U2;
        U3 = x.U3;
        z = x.z;
        
        if ~isfield(store, 'Z1')
            [~, store] = cost(x, store);
        end
        
        U1tZ1 = store.U1tZ1;
        U2tZ2 = store.U2tZ2;
        U3tZ3 = store.U3tZ3;
        
        Z1 = store.Z1;
        Z2 = store.Z2;
        Z3 = store.Z3;
        
        R1 = store.R1;
        R2 = store.R2;
        R3 = store.R3;
        
        U1tR1 = mymultfullsparse(U1', R1, mask1relative);
        U2tR2 = mymultfullsparse(U2', R2, mask2relative);
        U3tR3 = mymultfullsparse(U3', R3, mask3relative);
        
        gFac1 = myspmaskmult(U1, U1tR1, I1, J1relative);
        gFac2_1 = myspmaskmult(U2, U2tR2, I2_1, J2_1relative);
        gFac3_1 = myspmaskmult(U3, U3tR3, I3_1, J3_1relative);
        
        
        grad.z = lambda1*gFac1 + lambda2*gFac2_1 + lambda3*gFac3_1 ...
            + options.mu*z;
        
        grad.U1 = lambda1*(mymultsparsefull(R1, U1tZ1', mask1relative)...
            + mymultsparsefull(Z1, U1tR1', mask1relative));
        
        grad.U2 = lambda2*(mymultsparsefull(R2, U2tZ2', mask2relative)...
            + mymultsparsefull(Z2, U2tR2', mask2relative));
        
        grad.U3 = lambda3*(mymultsparsefull(R3, U3tZ3', mask3relative)...
            + mymultsparsefull(Z3, U3tR3', mask3relative));
        
        %         norm(x.z)
        %         norm(grad.z)
        %         %         norm(grad.U3, 'fro')
        %         norm(x.U3,'fro')
        %         norm(grad.U3,'fro')
        %         pause;
    end
    
    
    
    
    %% Euclidean Hessian
    problem.ehess = @ehess;
    function [ehess, store] = ehess(x, xdot, store)
        
        if ~isfield(store, 'Z1')
            [~, store] = cost(x, store);
        end
        
        % Basic computations
        U1 = x.U1;
        U2 = x.U2;
        U3 = x.U3;
        
        U1dot = xdot.U1;
        U2dot = xdot.U2;
        U3dot = xdot.U3;
        zdot = xdot.z;
        
        
        Z1 = store.Z1;
        Z2 = store.Z2;
        Z3 = store.Z3;
        
        Z1dot = zdot;
        Z2dot = Z1dot(mode1tomode2);
        Z3dot = Z1dot(mode1tomode3);
        
        
        U1tZ1 = store.U1tZ1;
        U2tZ2 = store.U2tZ2;
        U3tZ3 = store.U3tZ3;
        
        U1tZ1dot = mymultfullsparse(U1', Z1dot, mask1relative);
        U2tZ2dot = mymultfullsparse(U2', Z2dot, mask2relative);
        U3tZ3dot = mymultfullsparse(U3', Z3dot, mask3relative);
        
        U1dottZ1 = mymultfullsparse(U1dot', Z1, mask1relative);
        U2dottZ2 = mymultfullsparse(U2dot', Z2, mask2relative);
        U3dottZ3 = mymultfullsparse(U3dot', Z3, mask3relative);
        
        
        
        % R dot computation
        R1 = store.R1;
        R2 = store.R2;
        R3 = store.R3;
        
        
        Fac1dot = myspmaskmult([U1 U1dot], [U1tZ1dot + U1dottZ1 ; U1tZ1], I1, J1relative);
        Fac2_1dot = myspmaskmult([U2 U2dot], [U2tZ2dot + U2dottZ2  ; U2tZ2], I2_1, J2_1relative);
        Fac3_1dot = myspmaskmult([U3 U3dot], [U3tZ3dot + U3dottZ3 ;U3tZ3], I3_1, J3_1relative);
        
        
        Rdot = lambda1*Fac1dot + lambda2*Fac2_1dot + lambda3*Fac3_1dot;
        R1dot = Rdot;
        R2dot = Rdot(mode1tomode2);
        R3dot = Rdot(mode1tomode3);
        
        
        % Other computations
        U1tR1 = mymultfullsparse(U1', R1, mask1relative);
        U2tR2 = mymultfullsparse(U2', R2, mask2relative);
        U3tR3 = mymultfullsparse(U3', R3, mask3relative);
        
        
        U1tR1dot = mymultfullsparse(U1', R1dot, mask1relative);
        U2tR2dot = mymultfullsparse(U2', R2dot, mask2relative);
        U3tR3dot = mymultfullsparse(U3', R3dot, mask3relative);
        
        U1dottR1 = mymultfullsparse(U1dot', R1, mask1relative);
        U2dottR2 = mymultfullsparse(U2dot', R2, mask2relative);
        U3dottR3 = mymultfullsparse(U3dot', R3, mask3relative);
        
        
        
        gFac1dot = myspmaskmult([U1 U1dot], [U1tR1dot + U1dottR1 ; U1tR1], I1, J1relative);
        gFac2dot_1 = myspmaskmult([U2 U2dot], [U2tR2dot + U2dottR2 ; U2tR2], I2_1, J2_1relative);
        gFac3dot_1 = myspmaskmult([U3 U3dot], [U3tR3dot + U3dottR3 ; U3tR3], I3_1, J3_1relative);
        
        ehess.z = lambda1*gFac1dot + lambda2*gFac2dot_1 + lambda3*gFac3dot_1 ...
            + options.mu*zdot;
        
        ehess.U1 = lambda1*(mymultsparsefull(R1dot, U1tZ1', mask1relative)...
            + mymultsparsefull(R1, (U1tZ1dot + U1dottZ1)', mask1relative)...
            +mymultsparsefull(Z1dot, U1tR1', mask1relative) ...
            + mymultsparsefull(Z1, (U1tR1dot + U1dottR1)', mask1relative));
        
        
        
        ehess.U2 = lambda2*(mymultsparsefull(R2dot, U2tZ2', mask2relative)...
            + mymultsparsefull(R2, (U2tZ2dot + U2dottZ2)', mask2relative) ...
            + mymultsparsefull(Z2dot, U2tR2', mask2relative) ...
            + mymultsparsefull(Z2, (U2tR2dot + U2dottR2)', mask2relative));
        
        
        ehess.U3 = lambda3*(mymultsparsefull(R3dot, U3tZ3', mask3relative)...
            + mymultsparsefull(R3, (U3tZ3dot + U3dottZ3)', mask3relative)...
            + mymultsparsefull(Z3dot, U3tR3', mask3relative) ...
            + mymultsparsefull(Z3, (U3tR3dot + U3dottR3)', mask3relative));
        
    end
    
    
    
    
    %% Stats that we compute every iteration, useful in showing plots.
    
    options.statsfun = @mystatsfun;
    function stats = mystatsfun(problem, x, stats, store)
        U1 = x.U1;
        V1 = store.U1tZ1;
        U2 = x.U2;
        V2 = store.U2tZ2;
        U3 = x.U3;
        V3 = store.U3tZ3;
        
        A1_train = myspmaskmult(U1, V1, I1, J1relative);
        A_train = lambda1*A1_train(mode1totensor);
        B2_train = myspmaskmult(U2, V2, I2, J2relative);
        B_train = lambda2*B2_train(mode2totensor);
        C3_train = myspmaskmult(U3, V3, I3, J3relative);
        C_train = lambda3*C3_train(mode3totensor);
        error_train = A_train + B_train + C_train - Y;
        
        stats.loss_rmse = sqrt((error_train'*error_train)/length_entries); % RMSE on train data.
        
        if isfield(problem,'subs_test') && ~isempty(problem.subs_test)
            V1_full(:, J1unique) = V1;
            V2_full(:, J2unique) = V2;
            V3_full(:, J3unique) = V3;
            
            A1_test = myspmaskmult(U1, V1_full, I1_test, J1_test);
            A_test = lambda1*A1_test(converters_test.mode1totensor);
            B2_test = myspmaskmult(U2, V2_full, I2_test, J2_test);
            B_test = lambda2*B2_test(converters_test.mode2totensor);
            C3_test = myspmaskmult(U3, V3_full, I3_test, J3_test);
            C_test = lambda3*C3_test(converters_test.mode3totensor);
            
            error_test = A_test + B_test + C_test - Y_test;
            stats.loss_test_rmse = sqrt((error_test'*error_test)/length_entries_test); % RMSE on test data.
            
            if options.computeauc
                stats.loss_test_auc = mycomputeAUC(A_test + B_test + C_test, Y_test);
            end
            
            if options.verbosity > 1
                if options.computeauc
                    fprintf('LossRMSE %e TestLossRMSE %e  TestLossAUC %e\n', stats.loss_rmse, stats.loss_test_rmse, stats.loss_test_auc);
                else
                    fprintf('LossRMSE %e TestLossRMSE %e \n', stats.loss_rmse, stats.loss_test_rmse);
                    
                end
            end
        else
            if options.verbosity > 1
                fprintf('LossRMSE %e\n', stats.loss_rmse);
            end
        end
    end
    
    
    
    %% Additional stopping criteria.
    
    options.stopfun = @mystopfun;
    function stopnow = mystopfun(problem, x, info, last)
        stopnow = (last >= 3 && info(last).gradnorm/info(1).gradnorm < options.tolrelgradnorm);
    end
    
    
    
    
    %% Gradient checks
    %     checkgradient(problem);
    %     pause;
    %     checkhessian(problem);
    %     pause;
    
    
    
    %% Solver
    
    if strcmpi('TR', options.method)
        % Riemannian trustregions
        [Xopt,~,infos] = trustregions(problem, Uinit, options);
        
    elseif strcmpi('CG', options.method)
        % Riemannian conjugategradients
        options.beta_type = 'H-S';
        options.linesearch = @linesearch;
        options.ls_contraction_factor = 0.2;
        options.ls_optimism = 1.1;
        options.ls_suff_decr = 1e-4;
        options.ls_max_steps = 25;
        [Xopt,~,infos] = conjugategradient(problem, Uinit, options);
        
    end
    
    
    
    
    
end
