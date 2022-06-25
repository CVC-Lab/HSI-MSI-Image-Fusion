function IND = MNBS(Y,YN,n)
% Implementation of "A New Band Selection Method for Hyperspectral Image 
% Based on Data Quality"

S = Y-YN;
Cov_S = S.'*S;

Cov_N = YN.'*YN;

N = size(S,2);

k = N;

IND = 1:N; % index for selected bands

while k > n
    
    Q = zeros(1,k);
    
    for i = 1:k
        
        ind = setdiff(IND,IND(i));
        
        TEMP_S = Cov_S(ind,ind);
        [~,TEMP] = eig(TEMP_S);
        d_S = diag(TEMP);
        
        TEMP_N = Cov_N(ind,ind);
        [~,TEMP] = eig(TEMP_N);
        d_N = diag(TEMP);
        
        ratio = sum(log10(d_S./d_N));
        
        Q(i) = ratio;
        
    end
    
    [~,idx] = max(Q);
    
    fprintf('Among the %4d bands, removed band is %4d \n',k,IND(idx))
    
    IND = setdiff(IND,IND(idx)); % the one with low snr and high correlation is removed
    
    k = k-1;
    
end