function K = LHS_K(K0,D,FX,ind,IND,mu,tau,N1,N2,N3)

[p,q] = size(K0);

K = D.'*D*K0(:);
K = reshape(K,[p q]);
K = mu*K+(tau+mu)*K0;

temp = zeros(N1,N2);
temp(ind) = K0(:);
Fk = fft2(temp);

for band = 1:N3
    
    Fx = FX(:,band);
    Fx = reshape(Fx,[N1 N2]);
    
    temp = real(ifft2(Fx.*Fk));
    
    y = zeros(N1,N2);
    y(IND) = temp(IND);
    Fy = fft2(y);
    
    temp = real(ifft2(conj(Fx).*Fy));
    temp = temp(ind);
    
    temp = reshape(temp,[p q]);
    K = K+temp;
    
end