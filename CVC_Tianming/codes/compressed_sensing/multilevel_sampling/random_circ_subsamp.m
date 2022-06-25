% This function creates a sampling scheme as circles. Each circle should have 
% approximatly the same number of samples. The vector `b` defines the radius of 
% each circle. If the `include_border` is true, samples outside the outer circle
% will be included as an extra layer. `full_levels` defines how many of the first 
% levels which should be fully sampled. 
% 
% 
% INPUT:
% Im_rows        - Number of image rows.
% Im_cols        - Number of image cols.
% b              - Boundary between the different sampling layers.
% M              - Number of samples.
% full_levels    - Number of fully sampled levels (Optional. Default: 0).
% include_border - Bolean, if the outer circle should be included (Optional, Default: 0) 
%
% OUTPUT:
% pos - M × 2 matrix with the selected indices.
% 
function pos = random_circ_subsamp(Im_rows, Im_cols, b, M, full_levels, include_border)
     
    % test that size is a power of 2
    nu_m = log2(Im_rows);
    nu_n = log2(Im_cols);
     
    assert (abs(round(nu_m) - nu_m) < eps), 'Image size is not a power of 2';
    assert (abs(round(nu_n) - nu_n) < eps), 'Image size is not a power of 2'; 
    assert (nu_m == nu_n), 'Image must be square';
     
    if (nargin < 5) 
        full_levels = 0;
        include_border = 0;
    end
     
    if (nargin < 6)
        include_border = 0;
    end 
     
    sy = Im_rows/2;    % Center of circle vetically 
    sx = Im_cols/2;    % Center of circle horisontally 
    n = length(b);     % Number of bins 
    s = 1;             % Number of samples obtained so far
    pos = zeros(M,2);  % Samples
    k = 1;             % Current cirle level
    
    if (full_levels && pi*b(full_levels)^2 > M) 
        disp('Error: full_levels requier more that total amount of samples');
        return;
    end
    
    
    Y = zeros(Im_rows, Im_cols);
    k = 1;
    if (full_levels)
         
        r1 = 0;
        r2 = b(full_levels);
         
        l = round((sqrt(2)/2)*r1);
        r1sq = floor(r1^2);
        r2sq = ceil(r2^2);
         
        for x = l:r2
            for y = l:r2
                 
                if ((x^2 + y^2) >= r1sq && (x^2 + y^2 <= r2sq) ) 
                    
                    pos(s,:) = [sy + y, sx + x];
                    pos(s+1,:) = [sy + y, sx - x];
                    pos(s+2,:) = [sy - y, sx - x];
                    pos(s+3,:) = [sy - y, sx + x];
                    
                    Y(sy  + y, sx + x) = 1;
                    Y(sy  + y, sx - x) = 1;
                    Y(sy  - y, sx - x) = 1;
                    Y(sy  - y, sx + x) = 1;
                    
                    s = s + 4;
                     
                end
            end
        end
        k = k + full_levels;
    end
     
    if include_border
        l = n-k+2;
        bins = round(((M-s)/l)*ones(l,1)) - 1; 
        bins(end) = bins(end)/5;
    else 
        l = n-k+1;
        bins = round(((M-s)/l)*ones(l,1)) - 1; 
    end
     
     
     
    % Find number of samples within each level
    cnt = 0;
    while (s + sum(bins) <= M)  
        idx = 1 + mod(cnt,l);
        bins(idx) = bins(idx) + 1;
        cnt = cnt + 1;
    end 
     
    k1 = k;
    while(k <= n)    
         
        level = k - k1+1; 
         
        if (k == 1)
            r1 = 0;
            r2 = b(k);
        else 
            r1 = b(k-1);
            r2 = b(k);
        end 
        
        s_begin = s;
        failed = 1;
        while(s < s_begin + bins(level)); 
            r = r1 + (r2-r1)*rand(1)-1;
            theta = 2*pi*rand(1); 
             
            x = round(r*cos(theta));
            y = round(r*sin(theta));
            if (failed > 100)
                failed = 1;
                 
                bins(level) = bins(level) - 1;
                bins(level+1) = bins(level+1) + 1;
            end
            if (~Y(sy + y,sx + x))
                pos(s,:) = [sy+y, sx+x];
                s = s + 1;
                Y(sy + y,sx + x) = 1;
                failed = 1;
                %fprintf('s: %d, k: %d\n', s, k);
            else 
                failed = failed + 1;
            end
             
        end
        k = k + 1;
    end
    
    if include_border
        r1 = b(end);
        r2 = sqrt(Im_rows^2 + Im_cols^2);
         
        while(s <= M)
            r = r1 + (r2-r1)*rand(1);
            theta = 2*pi*rand(1); 
             
            x = round(r*cos(theta));
            y = round(r*sin(theta));
            if (abs(y) < Im_cols/2 && abs(x) < Im_rows/2)
                if (~Y(sy + y,sx + x))
                    pos(s,:) = [sy+y, sx+x];
                    s = s + 1;
                    Y(sy + y,sx + x) = 1;
                end
            end
        end 
    end
end 





