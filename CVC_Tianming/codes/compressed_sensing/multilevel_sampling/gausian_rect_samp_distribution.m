

function pos = gausian_rect_samp_distribution(Im_rows, Im_cols, nbr_samples,full_levels)
     
    % Test that size is a power of 2
    nu_m = log2(Im_rows);
    nu_n = log2(Im_cols);
     
    assert (abs(round(nu_m) - nu_m) < eps), 'Image size is not a power of 2';
    assert (abs(round(nu_n) - nu_n) < eps), 'Image size is not a power of 2'; 
    assert (nu_m == nu_n), 'Image must be square';
     
    Y = zeros(Im_rows, Im_cols); 
     
     
    bin_boundary = 2.^(0:nu_n); 
     
    pos = zeros(nbr_samples,2);
    s = 1;
     
    for i = 1:2^full_levels
        for j = 1:2^full_levels
            pos(s,:) = [i,j];
            s = s + 1;
        end
    end
     
    while s <= nbr_samples
        x = round(abs(Im_rows*randn(1,1)/3));
        y = round(abs(Im_rows*randn(1,1)/3));
        if (x < 1)
            x = round(2*(rand(1,1) + 1));
        end
        if (y < 1)
            y = round(2*(rand(1,1) + 1));
        end
        if (x > Im_rows)
            x = round((bin_boundary(nu_n) - bin_boundary(nu_n-1))*rand(1,1) + bin_boundary(nu_n-1));
        end
        if (y > Im_rows)
            y = round((bin_boundary(nu_n) - bin_boundary(nu_n-1))*rand(1,1) + bin_boundary(nu_n-1));
        end
        if (Y(x,y) < 1)
            Y(x,y) = Y(x,y) + 1;
            pos(s,:) = [x, y];
            s = s + 1;
        end
     end   
            
end


