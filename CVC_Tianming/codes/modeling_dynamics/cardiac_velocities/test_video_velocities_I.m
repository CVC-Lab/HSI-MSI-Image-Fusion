Vx = X(1:2:end,1:58);
Vy = X(2:2:end,1:58);

R = 114; % change here
M = 58;

SNR = zeros(2,R);

L = X(:,1:M)'*X(:,1:M);
l = diag(L);

%gamma = mean(sqrt(l));
%L = L/(gamma^2);
%L = 1+L;
%G = L(1:end-1,1:end-1).^2;
%AA = L(2:end,1:end-1).^2;
%AA = kron(AA,eye(2));
    
l = l';
D = repmat(l,[M 1])-2*L+repmat(l',[1 M]);
gamma = sum(sum(sqrt(D)))/(M*(M-1));
G = exp(-D(1:end-1,1:end-1)/(gamma^2));
AA = exp(-D(2:end,1:end-1)/(gamma^2));
AA = kron(AA,eye(2));

G = (G+G')/2;
[QQ,S] = eig(G);
ss = sqrt(diag(S));
[~,IND] = sort(ss,'descend');
ss = ss(IND);
ss = [ss ss];
ss = ss.';
ss = ss(:);
QQ = QQ(:,IND);
QQ = kron(QQ,eye(2));

G = kron(G,eye(2));

for r = 1:R
    
    Q = QQ(:,1:r);
    s = ss(1:r);
    
    Q = bsxfun(@times,Q,(1./s).');
    
    K = Q.'*AA*Q;
    [V,MU] = eig(K);
    mu = diag(MU);
    
    W = pinv(V);
    W = W.';
    
    XX = X(:,1:M-1);
    XX(1:53248,:) = X(1:2:end,1:M-1); % change here
    %XX(1,:) = X(1,1:M-1);
    XX(53249:end,:) = X(2:2:end,1:M-1); % change here
    %XX(2,:) = X(2,1:M-1);
    XX = reshape(XX,53248,[]); % change here
    %XX = reshape(XX,1,[]);
    
    XI = XX*Q*W;
    
    c = G(1:2,:)*(Q*V);
    c = c.';
    
    c1 = c(:,1);
    c2 = c(:,2);
    
    C1 = zeros(r,58);
    C2 = C1;
    
    for i = 1:58
        C1(:,i) = (mu.^(i-1)).*c1;
        C2(:,i) = (mu.^(i-1)).*c2;
    end
    
    X1 = XI*C1;
    SNR(1,r) = snr(Vx(:),Vx(:)-X1(:));
    %SNR(1,r) = snr(X(1,:),X(1,:)-X1);
    X2 = XI*C2;
    SNR(2,r) = snr(Vy(:),Vy(:)-X2(:));
    %SNR(2,r) = snr(X(2,:),X(2,:)-X2);
    
    disp(r)
    
end