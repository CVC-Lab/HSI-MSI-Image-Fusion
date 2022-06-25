clc; clear; close all

cd ..
cd test_data

load('Cardiac_CD_23.mat')
X = reshape(X,[],size(X,3));

cd ..
cd dynamic_mode_decomposition

r = 15; % rank of X

% exact DMD
tic;
XX = DMD(X,r);
t0 = toc;

err0 = norm(XX-X,'fro')^2/norm(X,'fro')^2;

% random DMD
k = r+10; % sketch size

tic;
XX = RDMD(X,r,k);
t1 = toc;

err1 = norm(XX-X,'fro')^2/norm(X,'fro')^2;

% EDMD with quadratic polynomial kernel
tic;
XX = EDMD_Q(X,r);
t2 = toc;

err2 = norm(XX-X,'fro')^2/norm(X,'fro')^2;

% EDMD with gaussian kernel
tic;
XX = EDMD_G(X,r);
t3 = toc;

err3 = norm(XX-X,'fro')^2/norm(X,'fro')^2;

% DMD performs well for this example, and RDMD accelerates the computation.
% EDMD_Q performs the best, meaning the underlying dynamic is not linear,
% and EDMD_G is a bit overkill.