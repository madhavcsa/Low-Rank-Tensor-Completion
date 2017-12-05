function Ax1 = computeLHS(x1, ...
        C, ...
        lambda1, lambda2, lambda3, ...
        U1, U2, U3, ...
        I1, I2_1, I3_1, ...
        J1, J2_1, J3_1,...
        mode1tomode2, mode1tomode3, ...
        mask1, mask2, mask3)
% [I2_1, J2_1] are mode 2 indices ordered according to mode 1.
% [I3_1, J3_1] are mode 3 indices ordered according to mode 1.
%     
% We compute Ax given x.
%     keyboard;
    U1tx = mymultfullsparse(U1', x1, mask1);
    U1U1tx = myspmaskmult(U1, U1tx, I1, J1);
    
    x2 = x1(mode1tomode2);
    U2tx = mymultfullsparse(U2', x2, mask2);
    U2U2tx_1 = myspmaskmult(U2, U2tx, I2_1, J2_1);
    
    x3 = x1(mode1tomode3);
    U3tx = mymultfullsparse(U3', x3, mask3);
    U3U3tx_1 = myspmaskmult(U3, U3tx, I3_1, J3_1);
    
    Ax1 = x1/(2*C) + lambda1*U1U1tx + lambda2*U2U2tx_1 + lambda3*U3U3tx_1;
end
