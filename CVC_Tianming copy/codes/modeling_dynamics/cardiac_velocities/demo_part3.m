% part 3 is about using EDMD to model vector fileds, the experiments are 
% conducted on the curl-free part

clc; clear; close all

load('C2.mat') % curl-free part, vector fields

X1 = C2(:,:,1:2:end);
X2 = C2(:,:,2:2:end);
clear C2
N = size(X1,1)*size(X1,2);
M = size(X1,3);
X1 = reshape(X1,[],M); % first component of the vector fields
X2 = reshape(X2,[],M); % second component of the vector fields
X = zeros(2*N,M);
X(1:2:end,:) = X1;
X(2:2:end,:) = X2;

r = 56; % use largest rank, since the data is hard to model 

% seperable quadratic polynomial kernel
[XX1,XX2] = EDMD_SQ(X,r);
err1(1) = norm(XX1-X1,'fro')^2/norm(X1,'fro')^2;
err1(2) = norm(XX2-X2,'fro')^2/norm(X2,'fro')^2;

% seperable gaussian polynomial kernel
[XX1,XX2] = EDMD_SG(X,r);
err2(1) = norm(XX1-X1,'fro')^2/norm(X1,'fro')^2;
err2(2) = norm(XX2-X2,'fro')^2/norm(X2,'fro')^2;

% curl-free kernel
[XX1,XX2] = EDMD_div(X,r);
err3(1) = norm(XX1-X1,'fro')^2/norm(X1,'fro')^2;
err3(2) = norm(XX2-X2,'fro')^2/norm(X2,'fro')^2;