


function pos = random_rect_subsamp(scheme)
    pos = 0;
    M = sum(scheme);
    
    nu = max(size(scheme));
    b = 2.^(0:nu);
    pos = zeros(M,2);
    
    if (scheme(1) == 4)
        pos(1,:) = [1,1];
        pos(2,:) = [1,2];
        pos(3,:) = [2,1];
        pos(4,:) = [2,2];
    else
        disp('First scheme element must be 4');
    end
    
    s = 5;
    k = 2;
    while ( (b(k+1)^2 - b(k)^2) == scheme(k) )
        
        l = b(k);
        h = b(k+1);
        
        for i = l+1:h
            for j = 1:h
                pos(s,:) = [i,j];
                s = s + 1;
            end
            for j = 1:l
                pos(s,:) = [j,i];
                s = s + 1;
            end
        end
        
        k = k + 1;
    end
    
    while( k <= nu)
        
        l = b(k);
        h = b(k+1);
        
        for t = 1:scheme(k)
            elem = round((h-1)*rand(1,2)+1);
            while(max(elem) <= l)
                elem = round((h-1)*rand(1,2)+1);
            end
            pos(s,:) = elem; 
            s = s + 1;
        end
        k = k + 1;
    end
    
end



