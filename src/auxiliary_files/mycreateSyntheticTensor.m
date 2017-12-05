function[subs, vals_noisy, substest, entriestest] = mycreateSyntheticTensor(tensorsize, tensorrank, OS, noise, tensortype, weights)
    
    
    n1 = tensorsize(1);
    n2 = tensorsize(2);
    n3 = tensorsize(3);
    
    r1 = tensorrank(1);
    r2 = tensorrank(2);
    r3 = tensorrank(3);
    
    
    dim = n1*r1 - r1^2 ...
        + n2*r2 - r2^2 ...
        + n3*r3 - r3^2 ...
        + r1*r2*r3;
    
    fraction = (OS*dim)/(n1*n2*n3);
    subs = mymakeOmegaSet(tensorsize, round(min(fraction, 1) * prod(tensorsize)));  % Training indices sorted as suggested by MS
    nentries = size(subs, 1); % Training number of known entries
    
    nentriestest = min(1*nentries, prod(tensorsize)); % Depends how big data can we handle.
    substest = mymakeOmegaSet(tensorsize, nentriestest);
    
    
    noise_vector = rand(nentries, 1);
    inverse_snr = noise;
    
    if strcmpi('Tucker', tensortype)
        
        %Generate data
        Xttensor = mymakeRandTensor(tensorsize, tensorrank); % Original tensor
        
        % Generate a training set
        vals = getValsAtIndex_mex(subs', Xttensor.G, Xttensor.U1', Xttensor.U2', Xttensor.U3'); % Training entries
        vals_noisy = vals + (inverse_snr * norm(vals) / norm(noise_vector))*noise_vector; % entries added with noise
        
        
        % Generate a test set
        entriestest = getValsAtIndex_mex(substest', Xttensor.G, Xttensor.U1', Xttensor.U2', Xttensor.U3');
        
        
    elseif strcmpi('Latent', tensortype)
        
        w1 = weights(1);
        w2 = weights(1);
        w3 = weights(1);
        
        
        % Generate a train set.
        [tensorindices, converters] = generate_indices(subs, tensorsize);
        
        AL = randn(n1, r1); AR = randn(r1, n2*n3);
        BL = randn(n2, r2); BR = randn(r2, n1*n3);
        CL = randn(n3, r3); CR = randn(r3, n1*n2);
        
        I1 = tensorindices.mode1.I;
        I2 = tensorindices.mode2.I;
        I3 = tensorindices.mode3.I;
        
        J1 = tensorindices.mode1.J;
        J2 = tensorindices.mode2.J;
        J3 = tensorindices.mode3.J;
        
        
        mode1totensor = converters.mode1totensor;
        mode2totensor = converters.mode2totensor;
        mode3totensor = converters.mode3totensor;
        
        valsA = w1*myspmaskmult(AL, AR, I1, J1);
        valsB = w2*myspmaskmult(BL, BR, I2, J2);
        valsC = w3*myspmaskmult(CL, CR, I3, J3);
        
        vals = valsA(mode1totensor) + valsB(mode2totensor) ...
            + valsC(mode3totensor);
        vals_noisy = vals + (inverse_snr * norm(vals) / norm(noise_vector))*noise_vector; % entries added with noise
        
        
        
        % Generate a test set.
        
        [tensorindices_test, converters_test] = generate_indices(substest, tensorsize);
        I1_test = tensorindices_test.mode1.I;
        I2_test = tensorindices_test.mode2.I;
        I3_test = tensorindices_test.mode3.I;
        
        J1_test = tensorindices_test.mode1.J;
        J2_test = tensorindices_test.mode2.J;
        J3_test = tensorindices_test.mode3.J;
        
        mode1totensor_test = converters_test.mode1totensor;
        mode2totensor_test = converters_test.mode2totensor;
        mode3totensor_test = converters_test.mode3totensor;
        
        valsA_test = w1*myspmaskmult(AL, AR, I1_test, J1_test);
        valsB_test = w2*myspmaskmult(BL, BR, I2_test, J2_test);
        valsC_test = w3*myspmaskmult(CL, CR, I3_test, J3_test);
        
        entriestest = valsA_test(mode1totensor_test) + valsB_test(mode2totensor_test) ...
            + valsC_test(mode3totensor_test);
        
    else
        
        error('Unknown tensortype. Should be "Tucker" or "Laten". \n');
        
    end
    
end