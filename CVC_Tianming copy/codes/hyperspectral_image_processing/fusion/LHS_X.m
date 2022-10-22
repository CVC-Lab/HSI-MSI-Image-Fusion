function X = LHS_X(X0,Fk,L,IND,N1,N2,N3,tau)

X = X0;

for band = 1:N3
    
    x = X0(:,band);
    
    x = reshape(x,[N1 N2]);
    Fx = fft2(x);
    
    x = real(ifft2(Fk.*Fx));
    
    temp = zeros(N1,N2);
    temp(IND) = x(IND);
    
    temp = fft2(temp);
    x = real(ifft2(conj(Fk).*temp));
    
    X(:,band) = x(:);
    
end

X = X+L*X0;
X = X+tau*X0;