clc; clear; close all

cd ..
cd test_data

load('YaleFace.mat')

cd ..
cd CUR

r = 20; % rank

tic;
[U,S,V] = svds(X,r);
t_opt = toc;

XX = U*S*V';
err_opt = norm(XX-X,'fro')^2/norm(X,'fro')^2; % optimal error

m = 81; % number of sampled columns, should be bigger than r
n = 81; % number of sampled rows, should be bigger than r

tic;
[C,U,R] = CUR_v1(X,r,m,n);
t_cur1 = toc;

XX = C*U*R;
err_cur1 = norm(XX-X,'fro')^2/norm(X,'fro')^2;

tic;
[C,U,R] = CUR_v2(X,r,m,n);
t_cur2 = toc;

XX = C*U*R;
err_cur2 = norm(XX-X,'fro')^2/norm(X,'fro')^2;