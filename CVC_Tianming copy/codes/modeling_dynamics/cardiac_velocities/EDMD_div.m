function [X1,X2] = EDMD_div(X,r)
% Modication of EDMD to model two observables together, using curl-free 
% kernel

% training
M = size(X,2);
N = size(X,1)/2;

DD = zeros(2*M,2*M);
for n = 1:N
    idx = 2*n-1:2*n;
    L = X(idx,1:M)'*X(idx,1:M);
    l = diag(L); 
    l = l';
    D = repmat(l,[M 1])-2*L+repmat(l',[1 M]);
    sigma = sum(sum(sqrt(D)))/(M*(M-1));
    for i = 1:M
        idx1 = (i-1)*2+1:i*2;
        x = X(idx,i);
        for j = 1:M
            idx2 = (j-1)*2+1:j*2;
            y = X(idx,j);
            c = norm(x-y)^2/(sigma^2);
            DD(idx1,idx2) = DD(idx1,idx2)+(1/sigma^2)*exp(-c/2)*(eye(2)-(x-y)*(x-y)'/(sigma^2));
        end
    end
end
DD = DD/N;

G = DD(1:2*(M-1),1:2*(M-1));
AA = DD(3:2*M,1:2*(M-1));

G = (G+G')/2;
[QQ,S] = eig(G);
ss = sqrt(diag(S));
[~,IND] = sort(ss,'descend');
ss = ss(IND);
QQ = QQ(:,IND);
Q = QQ(:,1:r);
s = ss(1:r);
Q = bsxfun(@times,Q,(1./s).');

K = Q.'*AA*Q;
[V,MU] = eig(K);
mu = diag(MU);

W = pinv(V);
W = W.';

XX = X(:,1:M-1);
XX(1:N,:) = X(1:2:end,1:M-1);
XX(N+1:end,:) = X(2:2:end,1:M-1);
XX = reshape(XX,N,[]);

XI = XX*Q*W;

% prediction
c = G(1:2,:)*(Q*V);
c = c.';

c1 = c(:,1);
c2 = c(:,2);

C1 = zeros(r,M);
C2 = C1;

for i = 1:M
    C1(:,i) = (mu.^(i-1)).*c1;
    C2(:,i) = (mu.^(i-1)).*c2;
end

X1 = XI*C1;
X2 = XI*C2;