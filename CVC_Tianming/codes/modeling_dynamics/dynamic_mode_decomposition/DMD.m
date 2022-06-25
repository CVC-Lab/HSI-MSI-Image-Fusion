function XX = DMD(X,r)
% "On dynamic mode decomposition: theory and applications"

% training
[U,S,V] = svd(X(:,1:end-1),'econ');
U = U(:,1:r);
s = diag(S(1:r,1:r));
V = V(:,1:r);

B = X(:,2:end)*V;
B = bsxfun(@times,B,(1./s)');
S = U'*B;

[W,Lambda] = eig(S);
lambda = diag(Lambda);

B = B*W;
W = bsxfun(@times,B,(1./lambda).');

% prediction
b = W\X(:,1);
        
M = size(X,2);
CC = zeros(r,M);

for i = 1:M
    CC(:,i) = (lambda.^(i-1)).*b;
end

XX = W*CC;