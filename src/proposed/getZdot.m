function [U1tz1dot, U2tz2dot, U3tz3dot, z1dot, z2dot, z3dot, ...
        U1dottz1, U2dottz2, U3dottz3] = getZdot(C, ...
        lambda1, lambda2, lambda3, ...
        U1, U2, U3, ...
        I1, I2_1, I3_1, ...
        J1, J2_1, J3_1, ...
        mode1tomode2, mode1tomode3, ...
        mask1, mask2, mask3, Y, pcgtol, pcgmaxiter, ...
        U1dot, U2dot, U3dot, ...
        z1, z2, z3, ...
        U1tz1, U2tz2, U3tz3)
% [I2_1, J2_1] are mode 2 indices ordered according to mode 1.
% [I3_1, J3_1] are mode 3 indices ordered according to mode 1.
% 
% Solve the dot system
% Axdot = Adot x for xdot, where A and x are known.
    
    U1dottz1 = mymultfullsparse(U1dot', z1, mask1);
    U2dottz2 = mymultfullsparse(U2dot', z2, mask2);
    U3dottz3 = mymultfullsparse(U3dot', z3, mask3);
    
    % Adotx computation, i.e.,
    % creating the RHS of the dot system.
    RHS1 = myspmaskmult([U1 U1dot], [U1dottz1; U1tz1], I1, J1);
    RHS2 = myspmaskmult([U2 U2dot], [U2dottz2; U2tz2], I2_1, J2_1);
    RHS3 = myspmaskmult([U3 U3dot], [U3dottz3; U3tz3], I3_1, J3_1);
    RHS = -lambda1*RHS1 - lambda2*RHS2 - lambda3*RHS3;
    
    % Axdot computation.

    myLHS = @(xdot) computeLHS(xdot, ...
        C, ...
        lambda1, lambda2, lambda3,...
        U1, U2, U3, ...
        I1, I2_1, I3_1, ...
        J1, J2_1, J3_1, ...
        mode1tomode2, mode1tomode3,...
        mask1, mask2, mask3);
    
    
    % Calling PCG for solving the dot system.
    [z1dot, flag, relres, iter, ~] = pcg(myLHS, RHS, pcgtol, pcgmaxiter);
    
    %     % Debug
    %     fprintf('flag: %d, ',flag);
    %     fprintf('iter: %d, ',iter);
    %     fprintf('relres: %d\n',relres);
    
    z2dot = z1dot(mode1tomode2);
    z3dot = z1dot(mode1tomode3);
    
    U1tz1dot = mymultfullsparse(U1', z1dot, mask1);
    U2tz2dot = mymultfullsparse(U2', z2dot, mask2);
    U3tz3dot = mymultfullsparse(U3', z3dot, mask3);
end


