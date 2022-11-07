function XX = EDMD_G(X,r)
% "A kernel-based method for data-driven Koopman spectral analysis"
% Kernel variant of EDMD, with gaussian kernel

% training
L = X'*X;
l = diag(L);

l = l';
m = size(X,2);
D = repmat(l,[m 1])-2*L+repmat(l',[1 m]);
gamma = sum(sum(sqrt(D)))/(m*(m-1));
G = exp(-D(1:end-1,1:end-1)/(gamma^2));
AA = exp(-D(2:end,1:end-1)/(gamma^2));

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