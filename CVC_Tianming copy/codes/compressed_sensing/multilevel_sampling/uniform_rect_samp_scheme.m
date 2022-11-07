% This function calculates how many of the samples each of the different 
% sampling layer should get.
% 
% Let a 2^R × 2^R  image requier a sampling rate of M samples, with   
% the boundaries between the different sampling layers equal to N^(0:R).
% This function will distribute the samples uniformly in a one dimensional 
% vector `m` whose `sum(3*m.^2) == M`. This vector `scheme =3*m.^2` is returned
% as the new sampling distribution. In cases where the number of samples M can not
% be written in this form, net necessary number of samples is added to `scheme`,
% to ensure `sum(scheme) == M`.   

function scheme = uniform_rect_samp_scheme(Im_rows, Im_cols, M)
    
    % test that size is a power of 2
    nu_m = log2(Im_rows);
    nu_n = log2(Im_cols);

    assert (abs(round(nu_m) - nu_m) < eps), 'Image size is not a power of 2';
    assert (abs(round(nu_n) - nu_n) < eps), 'Image size is not a power of 2'; 
    assert (nu_m == nu_n), 'Image must be square';
    nu = nu_n;
     
    m = ones(nu,1); 
    i=2;
    while(sum(m.^2) <= M/3) 
        m(i:nu) = m(i:nu) + 2^(i-1) - 2^(i-2);
        i = i + 1;
    end 
    i = i - 1;
    m(i:nu) = 2^(i-2);
    
    while(sum(3*m.^2) <= M)
        m(i:nu) = m(i:nu) + 1; 
    end
    m(i:nu) = m(i:nu) - 1;
    m = 3*m.^2;
    m(1) = 4;
    
    k = 0;
    while(sum(m) < M)
         s = i + mod(k,nu-i+1);
         m(s) = m(s) + 1;
         k = k + 1;
    end

    scheme = m;
end




