function XX = RDMD(X,r,k)
% "Randomized dynamic mode decomposition"

% training
[~,m] = size(X);

O = randn(m,k);
Y = X*O;
[Q,~] = qr(Y,0);

B = Q'*X;

[U,S,V] = svds(B(:,1:m-1),r);
s = diag(S);

BB = B(:,2:m)*(V*diag(1./s));
A = U'*BB;

[W,Lambda] = eig(A);
lambda = diag(Lambda);

W = Q*(BB*W);

% prediction
b = W\X(:,1);

M = size(X,2);
CC = zeros(r,M);

for i = 1:M
    CC(:,i) = (lambda.^(i-1)).*b;
end

XX = W*CC;