function b = HSI_MSI(x,P12,P3,S12,S3,n1,n2,n3,m12,m3,mode)
% forward and backward operator 

N = n1*n2;

if mode == 1
   
    X = reshape(x,N,n3);
   
   for j = 1:n3
       C = X(:,j);
       C = reshape(C,n1,n2);
       I = idct2(C);
       X(:,j) = I(:);
   end
   
   X = X';
   X = idct(X);
   
   X = X';
   Y = P12*X;
   Y = S12*Y(:);
   Z = X*P3;
   Z = S3*Z(:);
   
   b = [Y;Z];
   
else
    
    y = x(1:m12);
    y = S12'*y;
    Y = reshape(y,[],n3);
    Y = P12'*Y;
    
    z = x(m12+1:m12+m3);
    z = S3'*z;
    Z = reshape(z,N,[]);
    Z = Z*P3';
    
    X = Y+Z;
    
    for j = 1:n3
        C = X(:,j);
        C = reshape(C,n1,n2);
        I = dct2(C);
        X(:,j) = I(:);
    end
    
    X = X';
    X = dct(X);
    
    X = X';
    b = X(:);
    
end