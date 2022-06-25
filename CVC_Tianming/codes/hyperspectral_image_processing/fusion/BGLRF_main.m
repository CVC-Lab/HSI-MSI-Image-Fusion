function [X,K,err] = BGLRF_main(Y,Z,alpha,beta,radius) 
% X is initialized first using bicubic interpolation
% 
% Y: HSI in tensor format, between 0 and 1
% Z: MSI in tensor format, between 0 and 1
% alpha: regularization parameter for graph Laplacian
% beta: TV regularization parameter for the kernel
% radius: radius for assumed blur kernel
%
% X: recovered SRI
% K: estimated blur kernel
% err: relative changes in X and K
%
% Blind Hyperspectral-Multispectral Image Fusion via Graph Laplacian
% Regularization

%% Define parameters, etc.
[n1,n2,N3] = size(Y);
Y = reshape(Y,n1*n2,N3);

[N1,N2,~] = size(Z);

% graph Laplacian
GL = getLaplacian(Z);

tau = 0; % proximal weight
L = alpha*GL+tau*speye(N1*N2,N1*N2); % for updating X

kernel = (2*radius+1)*[1 1]; % search space for blur kernel
D = nabla(kernel(1),kernel(2)); % operator for partial derivatives

center = zeros(1,2);
if mod(N1,2)
    center(1) = (N1+1)/2;
else
    center(1) = N1/2+1;
    if mod(N2,2)
        center(2) = (N2+1)/2;
    else
        center(2) = N2/2+1;
    end
end

I = (center(1)-radius):(center(1)+radius);
I = I.';
I = repmat(I,[1 2*radius+1]);
I = I(:);
J = (center(2)-radius):(center(2)+radius);
J = repmat(J,[2*radius+1 1]);
J = J(:);
ind = sub2ind([N1 N2],I,J); % indices for blur kernel

ratio = round(N1/n1); % downsampling ratio

I = 2:ratio:N1;
I = I.';
I = repmat(I,[1 n2]);
I = I(:);
J = 2:ratio:N2;
J = repmat(J,[n1 1]);
J = J(:);
IND = sub2ind([N1 N2],I,J); % indices for HSI

%% Initialization
FY = zeros(N1*N2,N3); % used in all the iterations
X = zeros(N1*N2,N3);

for band = 1:N3
    y = Y(:,band);
    y = reshape(y,[n1 n2]);
    temp = zeros(N1,N2);
    temp(IND) = y;
    temp = fft2(temp);
    FY(:,band) = temp(:);
    x = imresize(y,[N1 N2]); 
    X(:,band) = x(:);
end

K = zeros(kernel);

maxiters = 100; % maximum number of iterations
err = zeros(maxiters,2);

%% Proximal alternating minization
for iter = 1:maxiters

    % update K
    FX = zeros(N1*N2,N3);
    RHS = size(kernel(1)*kernel(2),1);
    
    for band = 1:N3
        x = X(:,band);
        x = reshape(x,[N1 N2]);
        Fx = fft2(circshift(x,1-center));
        FX(:,band) = Fx(:);
        Fy = FY(:,band);
        Fy = reshape(Fy,[N1 N2]);
        y = real(ifft2(conj(Fx).*Fy));
        RHS = RHS+y(ind);
    end
    
    RHS = reshape(RHS,kernel(1),kernel(2));
    RHS = RHS+tau*K;
    
    K_pre = K;
    K = ADMM_K(RHS,K,D,FX,ind,IND,beta,tau,N1,N2,N3);
    
    % update X    
    KK = zeros(N1,N2);
    KK(ind) = K;
    Fk = fft2(circshift(KK,1-center));
    
    RHS = zeros(N1*N2,N3);
    for band = 1:N3
        temp = FY(:,band);
        temp = reshape(temp,[N1 N2]);
        temp = real(ifft2(conj(Fk).*temp));
        RHS(:,band) = temp(:);
    end
    RHS = RHS+tau*X;
    
    X_pre = X;
    X = CG_X(RHS,X,Fk,L,IND,N1,N2,N3,tau);
    
    err(iter,1) = norm(K-K_pre,'fro')/norm(K,'fro');
    err(iter,2) = norm(X-X_pre,'fro')/norm(X,'fro');
    
    fprintf('GLRF - iteration %3d: change.K = %.4f, change.X = %.4f \n',iter,err(iter,1),err(iter,2))
    
    if err(iter,2) < 1e-3
        err = err(1:iter,:);
        X = reshape(X,N1,N2,N3);
        break
    end
    
end