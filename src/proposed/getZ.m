function [U1tz1, U2tz2, U3tz3, z1, z2, z3] = getZ(C, ...
        lambda1, lambda2, lambda3,...
        U1, U2, U3, ...
        I1, I2_1, I3_1, ...
        J1, J2_1, J3_1, ...
        mode1tomode2, mode1tomode3, ...
        mask1, mask2, mask3, ...
        Y, pcgtol, pcgmaxiter)
% [I2_1, J2_1] are mode 2 indices ordered according to mode 1.
% [I3_1, J3_1] are mode 3 indices ordered according to mode 1.
%
% We want to solve Ax = b.

    % Ax computation.
    myLHS = @(x) computeLHS(x, ...
        C, ...
        lambda1, lambda2, lambda3,...
        U1, U2, U3, ...
        I1, I2_1, I3_1, ...
        J1, J2_1, J3_1, ...
        mode1tomode2, mode1tomode3,...
        mask1, mask2, mask3);
    
    % Call PCG. 
    [z1,flag,relres,iter,~] = pcg(myLHS, Y, pcgtol, pcgmaxiter);
%     [z1,flag,relres,iter,~] = pcg(myLHS, Y, 1e-16, 1000);
    
    
%         % Debug
%         fprintf('       flag: %d, ',flag);
%         fprintf('iter: %d, ',iter);
%         fprintf('relres: %d\n',relres);

    z2 = z1(mode1tomode2);
    z3 = z1(mode1tomode3);
    
    U1tz1 = mymultfullsparse(U1', z1, mask1);
    U2tz2 = mymultfullsparse(U2', z2, mask2);
    U3tz3 = mymultfullsparse(U3', z3, mask3);
end




