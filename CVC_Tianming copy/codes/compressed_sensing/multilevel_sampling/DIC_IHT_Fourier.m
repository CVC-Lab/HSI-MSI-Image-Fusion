function f = DIC_IHT_Fourier(y,IND,m,n1,n2,level)
% iterative hard thresholding for images sparse in the wavelet domain, measurement matrix is Fourier
% the total sparsity [m] is required
%
% an implementation of the algorithm in 
% [Dictionary-Sparse Recovery via Thresholding-Based Algorithms]

f = zeros(n1,n2);
temp = -y;

obj(1) = norm(temp)^2;

for iter = 1:9999
    
    g = zeros(n1,n2);
    g(IND) = temp;
    g = ifft2(ifftshift(g))*sqrt(n1*n2);
    
    temp = fftshift(fft2(g))/sqrt(n1*n2);
    alpha = norm(g,'fro')^2/(norm(temp(IND))^2);
    
    w = f-alpha*g;
    
    [C,S] = wavedec2(w,level,'haar');
    [~,ind] = sort(abs(C),'descend');
    CC = C;
    C = zeros(size(CC));
    C(ind(1:m)) = CC(ind(1:m));
    
    f = waverec2(C,S,'haar');
    
    TEMP = fftshift(fft2(f));
    temp = TEMP(IND)/(sqrt(n1*n2))-y;
        
    obj(iter+1) = norm(temp)^2;
    rel(iter) = abs(obj(iter+1)-obj(iter))/obj(iter);
    
    if rel(iter) < 1e-3
        break;
    else
        fprintf('iteration %4d: obj = %e & rel = %.4f \n',iter,obj(iter+1),rel(iter))
    end
    
end