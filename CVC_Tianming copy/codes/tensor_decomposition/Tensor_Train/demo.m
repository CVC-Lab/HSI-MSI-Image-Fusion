clc; clear; close all

cd ..
cd test_data

load('Cardiac_DE_2.mat')
norm_X = norm(X(:))^2;

cd ..
cd Tensor_Train

% third order tensors, compute partial SVD twice (with proper reshaping) 
r1 = 10; r2 = 20;

[G1,G2,G3] = TT(X,r1,r2); % one can replace the svds in TT.m with other low
% rank approximation algorithms

XX = TT_inv(G1,G2,G3);
norm_XX = norm(XX(:))^2;

inner = TT_innerprod(G1,G2,G3,X);
err = (norm_X+norm_XX-2*inner)/norm_X; 