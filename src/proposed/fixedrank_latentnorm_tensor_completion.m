function[Xopt, infos, options] = fixedrank_latentnorm_tensor_completion(problem, Uinit, options)
    
    %% Problem description
    
    C = problem.C;
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
    
    
    
    
    
    
    %% Generate indices for the given data.
    
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
        
        R1_test = zeros(r1, size(mask1, 2));
        R2_test = zeros(r2, size(mask2, 2));
        R3_test = zeros(r3, size(mask3, 2));
        
    end
    
    
    
    
    %% Local defaults for options
    
    localdefaults.maxiter = 50; % Max iterations.
    localdefaults.tolgradnorm = 1e-6; % Absolute tolerance on Gradnorm.
    localdefaults.maxinner = 30; % Max inner iterations for the tCG step.
    localdefaults.pcgtol = 1e-5; % Tolerance for solving the linear system.
    localdefaults.pcgmaxiter = 50; % Max PCG iterations for solving the linear system.
    localdefaults.dotpcgtol = 1e-5; % Tolerance for solving the dot linear system.
    localdefaults.dotpcgmaxiter = 20; % Max PCG iterations for solving the dot linear system.
    localdefaults.tolrelgradnorm = 1e-8; % Gradnorm/initGradnorm tolerance.
    localdefaults.method = 'TR'; % Default solver is trustregions (TR).
    localdefaults.verbosity = 2; % Show output.
    localdefaults.computeauc = false; % Compute the AUC, useful in link prediction applications.
    
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    
    
    
    %% Initialization
    
    if ~exist('Uinit', 'var') % Uinit is a structure.
        Uinit = [];
    end
    
    
    
    
    %% Manifold geometry: the Cartesian product of three spectrahedron manifolds.
    
    mymanifold.U1 = spectrahedronfactory(n1, r1);
    mymanifold.U2 = spectrahedronfactory(n2, r2);
    mymanifold.U3 = spectrahedronfactory(n3, r3);
    problem.M = productmanifold(mymanifold);
    
    
    
    
    %% Cost
    
    problem.cost = @cost;
    function [f, store] = cost(U, store)
        if ~isfield(store, 'Z1')
            % Solve for Z given U.
            [U1tZ1, U2tZ2, U3tZ3, Z1, Z2, Z3] = getZ(C, ...
                lambda1, lambda2, lambda3, ...
                U.U1, U.U2, U.U3, ...
                I1, I2_1, I3_1, ...
                J1relative, J2_1relative, J3_1relative, ...
                mode1tomode2, mode1tomode3,...
                mask1relative, mask2relative, mask3relative,...
                Y1, options.pcgtol, options.pcgmaxiter);
            store.U1tZ1 = U1tZ1;
            store.U2tZ2 = U2tZ2;
            store.U3tZ3 = U3tZ3;
            store.Z1 = Z1;
            store.Z2 = Z2;
            store.Z3 = Z3;
        end
        Z1 = store.Z1;
        U1tZ1 = store.U1tZ1;
        U2tZ2 = store.U2tZ2;
        U3tZ3 = store.U3tZ3;
        
        f = (Y1'*Z1) - (Z1'*Z1)/(4*C) - 0.5*( lambda1*norm(U1tZ1,'fro')^2  + lambda2*norm(U2tZ2,'fro')^2 + lambda3*norm(U3tZ3,'fro')^2 );
    end
    
    
    
    
    %% Euclidean gradient
    
    problem.egrad = @egrad;
    function [grad, store] = egrad(U, store)
        if ~isfield(store, 'Z1')
            [~, store] = cost(U, store);
        end
        U1tZ1 = store.U1tZ1;
        U2tZ2 = store.U2tZ2;
        U3tZ3 = store.U3tZ3;
        Z1 = store.Z1;
        Z2 = store.Z2;
        Z3 = store.Z3;
        
        grad.U1 = - lambda1*mymultsparsefull(Z1, U1tZ1', mask1relative);
        grad.U2 = - lambda2*mymultsparsefull(Z2, U2tZ2', mask2relative);
        grad.U3 = - lambda3*mymultsparsefull(Z3, U3tZ3', mask3relative);
        
    end
    
    
    
    %% Euclidean Hessian
    
    problem.ehess = @ehess;
    function [graddot, store] = ehess(U, Udot, store)
        if ~isfield(store, 'Z1')
            [~, store] = cost(U, store);
        end
        U1tZ1 = store.U1tZ1;
        U2tZ2 = store.U2tZ2;
        U3tZ3 = store.U3tZ3;
        Z1 = store.Z1;
        Z2 = store.Z2;
        Z3 = store.Z3;
        
        % Solve for Zdot given U, Udot, and Z.
        [U1tZ1dot, U2tZ2dot, U3tZ3dot,Z1dot, Z2dot, Z3dot, ...
            U1dottZ1, U2dottZ2, U3dottZ3] = getZdot(C,...
            lambda1, lambda2, lambda3,...
            U.U1, U.U2, U.U3, ...
            I1, I2_1, I3_1, ...
            J1relative, J2_1relative, J3_1relative, ...
            mode1tomode2, mode1tomode3, ...
            mask1relative, mask2relative, mask3relative, ...
            Y1, options.dotpcgtol, options.dotpcgmaxiter,...
            Udot.U1, Udot.U2, Udot.U3, ...
            Z1, Z2, Z3, ...
            U1tZ1, U2tZ2, U3tZ3);
        
        
        graddot.U1 = - lambda1*(mymultsparsefull(Z1dot, U1tZ1', mask1relative) ...
            + mymultsparsefull(Z1, U1dottZ1' + U1tZ1dot', mask1relative));
        graddot.U2 = - lambda2*(mymultsparsefull(Z2dot, U2tZ2', mask2relative) ...
            + mymultsparsefull(Z2, U2dottZ2' + U2tZ2dot', mask2relative));
        graddot.U3 = - lambda3*(mymultsparsefull(Z3dot, U3tZ3', mask3relative) ...
            + mymultsparsefull(Z3, U3dottZ3' + U3tZ3dot', mask3relative));
        
    end
    
    
    
    
    %% Stats that we compute every iteration, useful in showing plots.
    
    options.statsfun = @mystatsfun;
    function stats = mystatsfun(problem, U, stats, store)
        L1 = U.U1;
        R1 = store.U1tZ1;
        L2 = U.U2;
        R2 = store.U2tZ2;
        L3 = U.U3;
        R3 = store.U3tZ3;
        
        A1_train = myspmaskmult(L1, R1, I1, J1relative);
        A_train = lambda1*A1_train(mode1totensor);
        B2_train = myspmaskmult(L2, R2, I2, J2relative);
        B_train = lambda2*B2_train(mode2totensor);
        C3_train = myspmaskmult(L3, R3, I3, J3relative);
        C_train = lambda3*C3_train(mode3totensor);
        error_train = A_train + B_train + C_train - Y;
        stats.loss_rmse = sqrt((error_train'*error_train)/length_entries); % RMSE on train data.
        
        if isfield(problem,'subs_test') && ~isempty(problem.subs_test)
            R1_test(:, J1unique) = R1;
            R2_test(:, J2unique) = R2;
            R3_test(:, J3unique) = R3;
            
            A1_test = myspmaskmult(L1, R1_test, I1_test, J1_test);
            A_test = lambda1*A1_test(converters_test.mode1totensor);
            B2_test = myspmaskmult(L2, R2_test, I2_test, J2_test);
            B_test = lambda2*B2_test(converters_test.mode2totensor);
            C3_test = myspmaskmult(L3, R3_test, I3_test, J3_test);
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
    function stopnow = mystopfun(problem, U, info, last)
        stopnow = (last >= 3 && info(last).gradnorm/info(1).gradnorm < options.tolrelgradnorm);
    end
    
    
    
    
    %% Gradient and Hessian checks
    
    %     checkgradient(problem);
    %     pause;
    %     checkhessian(problem);
    %     pause;
    
    
    
    %% Solver
    
    if strcmpi('TR', options.method)
        % Riemannian trustregions
        [Uopt,~,infos] = trustregions(problem, Uinit, options);
    
    elseif strcmpi('CG', options.method)
        % Riemannian conjugategradients
        options.beta_type = 'H-S';
        options.linesearch = @linesearch;
        options.ls_contraction_factor = 0.2;
        options.ls_optimism = 1.1;
        options.ls_suff_decr = 1e-4;
        options.ls_max_steps = 25;
        [Uopt,~,infos] = conjugategradient(problem, Uinit, options);
        
    end
    
    
    %% Compute optimal Zopt given Uopt.
    
    [U1tZ1opt, U2tZ2opt, U3tZ3opt, Z1opt, Z2opt, Z3opt] = getZ(C, ...
        lambda1, lambda2, lambda3, ...
        Uopt.U1, Uopt.U2, Uopt.U3, ...
        I1, I2_1, I3_1, ...
        J1relative, J2_1relative, J3_1relative, ...
        mode1tomode2, mode1tomode3, ...
        mask1relative, mask2relative, mask3relative, ...
        Y1, 1e-10, 100);
    
    
    
    
    %% Full solution structure
    Xopt.U1 = Uopt.U1;
    Xopt.U2 = Uopt.U2;
    Xopt.U3 = Uopt.U3;
    Xopt.U1tZ1 = U1tZ1opt;
    Xopt.U2tZ2 = U2tZ2opt;
    Xopt.U3tZ3 = U3tZ3opt;
    Xopt.Z1 = Z1opt;
    Xopt.Z2 = Z2opt;
    Xopt.Z3 = Z3opt;
    
    
    
    %%
end