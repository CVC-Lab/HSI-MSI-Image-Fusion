function [X1,X2] = EDMD_SQ(X,r)
% Modication of EDMD to model two observables together, using seperable
% quadratic polynomial kernel

% training
L = X'*X;
l = diag(L);

gamma = mean(sqrt(l));
L = L/(gamma^2);
L = 1+L;
G = L(1:end-1,1:end-1).^2;
AA = L(2:end,1:end-1).^2;
AA = kron(AA,eye(2));

G = (G+G')/2; 
[QQ,S] = eig(G);
G = kron(G,eye(2));

ss = sqrt(diag(S));
[~,IND] = sort(ss,'descend');
ss = ss(IND);
ss = [ss ss];
ss = ss.';
ss = ss(:);
QQ = QQ(:,IND);
QQ = kron(QQ,eye(2));
Q = QQ(:,1:r);
s = ss(1:r);
Q = bsxfun(@times,Q,(1./s).');

K = Q.'*AA*Q;
[V,MU] = eig(K);
mu = diag(MU);

W = pinv(V);
W = W.';

XX = X(:,1:end-1);
N = size(X,1)/2;
XX(1:N,:) = X(1:2:end,1:end-1);
XX(N+1:end,:) = X(2:2:end,1:end-1);
XX = reshape(XX,N,[]);

XI = XX*Q*W;

% prediction
c = G(1:2,:)*(Q*V);
c = c.';

c1 = c(:,1);
c2 = c(:,2);

M = size(X,2);
C1 = zeros(r,M);
C2 = C1;

for i = 1:M
    C1(:,i) = (mu.^(i-1)).*c1;
    C2(:,i) = (mu.^(i-1)).*c2;
end

X1 = XI*C1;
X2 = XI*C2;