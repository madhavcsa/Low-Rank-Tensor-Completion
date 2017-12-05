function [tensorindices, converters]= generate_indices(subs, tensorsize)
    I = subs(:,1);
    J = subs(:,2);
    K = subs(:,3);
    n1 = tensorsize(1);
    n2 = tensorsize(2);
    n3 = tensorsize(3);
    
    
    % mode-1 unfolding information
    temp1 = I + n1*(J-1) + n1*n2*(K-1);
    [mode_1_idx, mode_1_converter] = sort(temp1);
    mode_1_converter_inverse(mode_1_converter) = 1:length(mode_1_converter);
    d1 = n1;
    T1 = n2*n3;
    [I1,J1] = ind2sub([d1,T1],mode_1_idx);
    I1 = uint32(I1);
    J1 = uint32(J1);
    mask1 = sparse(double(I1), double(J1), ones(length(I1), 1), d1, T1, length(I1));
    [J1_org_unique, J1_occ, J1_pseudo] = unique(J1);
    J1_pseudo = uint32(J1_pseudo);
    mask1_pseudo = sparse(double(I1), double(J1_pseudo), ones(length(I1), 1), d1, double(J1_pseudo(end)), length(I1));
    
    
    tensorindices.mode1.I = I1;
    tensorindices.mode1.J = J1;
    tensorindices.mode1.mask = mask1;
    tensorindices.mode1.Junique = J1_org_unique;
    tensorindices.mode1.Jrelative = J1_pseudo;
    tensorindices.mode1.Jocc = uint32(J1_occ-1);
    tensorindices.mode1.maskrelative = mask1_pseudo;
    
    
    
    % mode-2 unfolding information
    temp2 = J + n2*(I-1) + n1*n2*(K-1);
    [mode_2_idx, mode_2_converter] = sort(temp2);
    mode_2_converter_inverse(mode_2_converter) = 1:length(mode_2_converter);
    d2 = n2;
    T2 = n1*n3;
    [I2,J2] = ind2sub([d2,T2],mode_2_idx);
    I2 = uint32(I2);
    J2 = uint32(J2);
    mask2 = sparse(double(I2), double(J2), ones(length(I2), 1), d2, T2, length(I2));
    [J2_org_unique,J2_occ,J2_pseudo] = unique(J2);
    J2_pseudo = uint32(J2_pseudo);
    mask2_pseudo = sparse(double(I2), double(J2_pseudo), ones(length(I2), 1), d2, double(J2_pseudo(end)), length(I2));
    
    tensorindices.mode2.I = I2;
    tensorindices.mode2.J = J2;
    tensorindices.mode2.mask = mask2;
    tensorindices.mode2.Junique = J2_org_unique;
    tensorindices.mode2.Jrelative = J2_pseudo;
    tensorindices.mode2.Jocc = uint32(J2_occ-1);
    tensorindices.mode2.maskrelative = mask2_pseudo;
    
    
    
    % mode-3 unfolding information
    temp3 = K + n3*(I-1) + n1*n3*(J-1);
    [mode_3_idx, mode_3_converter] = sort(temp3);
    mode_3_converter_inverse(mode_3_converter) = 1:length(mode_3_converter);
    d3 = n3;
    T3 = n1*n2;
    [I3,J3] = ind2sub([d3,T3],mode_3_idx);
    I3 = uint32(I3);
    J3 = uint32(J3);
    mask3 = sparse(double(I3), double(J3), ones(length(I3), 1), d3, T3, length(I3));
    [J3_org_unique,J3_occ,J3_pseudo] = unique(J3);
    J3_pseudo = uint32(J3_pseudo);
    mask3_pseudo = sparse(double(I3), double(J3_pseudo), ones(length(I3), 1), d3, double(J3_pseudo(end)), length(I3));
    
    tensorindices.mode3.I = I3;
    tensorindices.mode3.J = J3;
    tensorindices.mode3.mask = mask3;
    tensorindices.mode3.Junique = J3_org_unique;
    tensorindices.mode3.Jrelative = J3_pseudo;
    tensorindices.mode3.Jocc = uint32(J3_occ-1);
    tensorindices.mode3.maskrelative = mask3_pseudo;
    
    temp = 1:length(I);
    mode_21_converter = temp(mode_2_converter_inverse);
    mode_21_converter = mode_21_converter(mode_1_converter);
    
    mode_31_converter = temp(mode_3_converter_inverse);
    mode_31_converter = mode_31_converter(mode_1_converter);
    
    mode_12_converter = temp(mode_1_converter_inverse);
    mode_12_converter = mode_12_converter(mode_2_converter);
    
    mode_13_converter = temp(mode_1_converter_inverse);
    mode_13_converter = mode_13_converter(mode_3_converter);
    
    % Indices coverters
    converters.tensortomode1 = mode_1_converter;
    converters.tensortomode2 = mode_2_converter;
    converters.tensortomode3 = mode_3_converter;
    
    converters.mode1totensor = mode_1_converter_inverse;
    converters.mode2totensor = mode_2_converter_inverse;
    converters.mode3totensor = mode_3_converter_inverse;
    converters.mode2tomode1 = mode_21_converter ;
    converters.mode3tomode1 = mode_31_converter;
    converters.mode1tomode2 = mode_12_converter;
    converters.mode1tomode3 = mode_13_converter;
end