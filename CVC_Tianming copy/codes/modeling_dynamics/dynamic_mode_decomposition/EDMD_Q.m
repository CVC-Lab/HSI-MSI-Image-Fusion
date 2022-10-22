function XX = EDMD_Q(X,r)
% "A kernel-based method for data-driven Koopman spectral analysis"
% Kernel variant of EDMD, with quadratic polynomial kernel

% training
L = X'*X;
l = diag(L);

gamma = mean(sqrt(l));
L = L/(gamma^2);
L = 1+L;
G = L(1:end-1,1:end-1).^2;
AA = L(2:end,1:end-1).^2;

G = (G+G')/2;
[Q,S] = eigs(G,r,'la');

s = sqrt(diag(S));
Q = bsxfun(@times,Q,(1./s).');

K = Q.'*AA*Q;
[V,MU] = eig(K);
mu = diag(MU);

W = pinv(V);
W = W.';
XI = X(:,1:end-1)*Q*W;

% prediction
c = G(1,:)*(Q*V);
c = c.';

M = size(X,2);
CC = zeros(r,M);

for i = 1:M
    cc = (mu.^(i-1)).*c;
    CC(:,i) = cc;
end

XX = XI*CC;