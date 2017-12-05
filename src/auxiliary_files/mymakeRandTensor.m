function X = mymakeRandTensor( n, k )
%MAKERANDTENSOR Create a random Tucker tensor
%   X = MAKERANDTENSOR( N, K ) creates a random Tucker tensor X stored as a 
%   ttensor object. The entries of the core tensor the basis factors are chosen
%   independently from the uniform distribution on [0,1]. Finally, the basis factors
%   are orthogonalized using a QR procedure.
%
%   See also makeOmegaSet
%

%   GeomCG Tensor Completion. Copyright 2013 by
%   Michael Steinlechner
%   Questions and contact: michael.steinlechner@epfl.ch
%   BSD 2-clause license, see LICENSE.txt
%
%    Modified by BM to be independent of tensor toolbox, 2015.

    [U1,R1] = qr( rand( n(1), k(1) ), 0);
    [U2,R2] = qr( rand( n(2), k(2) ), 0);
    [U3,R3] = qr( rand( n(3), k(3) ), 0);

    C  = rand(k); % Modified: replace tenrand( k );
    
    % Lines by MS
    % C = ttm( C, {R1,R2,R3},[1,2,3]);
    % X = ttensor( C, {U1, U2, U3} );
    
    % Modified
    r1 = k(1);
    r2 = k(2);
    r3 = k(3);
    
    % Multplication by R1
    C1 = reshape(C, r1, r2*r3);
    CR1 = reshape(R1*C1, r1, r2, r3);
    
    % Multplication by R2
    C2 = reshape(permute(CR1, [2 1 3]), r2, r1*r3); 
    CR1R2 = permute(reshape(R2*C2, r2, r1, r3), [2 1 3]);
    
    % Multplication by R3
    C3 = reshape(permute(CR1R2, [3 1 2]), r3, r1*r2);  
    
    
    CR1R2R3 = permute(reshape(R3*C3, r3, r1, r2), [2 3 1]); 
        
    X.U1 = U1;
    X.U2 = U2;
    X.U3 = U3;
    X.G = CR1R2R3;
    
end
