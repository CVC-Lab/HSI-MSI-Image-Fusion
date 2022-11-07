% this demo demonstrates the idea of compressed sensing based fusion on a
% simple example, the sparsifying system is the kronecker product of 1D DCT
% (for the spectral) and 2D DCT (for the spatial)

clc; clear; close all
addpath('../spgl1-1.9')

load('lena.mat')
X = I; % SRI is a RGB image
clear I
[n1,n2,n3] = size(X);
X = reshape(X,[],n3);

% generate HSI
ratio = 2; % downsampling ratio of HSI
radius = 2*ratio+1;
psf = fspecial('gaussian',[radius radius],ratio/(2.355));

bccb = generate_bccb(n1,n2,psf);
I = zeros(n1,n2);
I(1:ratio:end,1:ratio:end) = 1;
ind = I == 1;
P12 = bccb(ind,:);

Y = P12*X;

% generate MSI
P3 = [0.2990;0.5870;0.1140];
Z = X*P3;

% compressed measurements of HSI
M12 = length(Y(:));
p12 = 0.5; % sampling ratio
m12 = floor(p12*M12);
S12 = sqrt(1/m12)*randn(m12,M12); % measurement matrix of HSI
Y = S12*Y(:);

% compressed measurements of MSI
M3 = length(Z(:));
p3 = 0.5; % sampling ratio
m3 = floor(p3*M3);
S3 = sqrt(1/m3)*randn(m3,M3); % measurement matrix of MSI
Z = S3*Z(:);

b = [Y;Z];

% recovery based on l1 minimization
opA = @(x,mode) HSI_MSI(x,P12,P3,S12,S3,n1,n2,n3,m12,m3,mode);
XX = spg_bp(opA,b); % DCT coefficients

XX = reshape(XX,[],n3);
for j = 1:n3
    C = XX(:,j);
    C = reshape(C,n1,n2);
    I = idct2(C);
    XX(:,j) = I(:);
end
XX = XX';
XX = idct(XX);
XX = XX';

err = norm(XX-X,'fro')^2/norm(X,'fro')^2;

rmpath('../spgl1-1.9')