clc; clear; close all

cd ..
cd test_data

load('YaleFace.mat')

cd ..
cd SketchySVD

r = 20; % rank
k = 4*r+1;

[Q,s,P,cpu] = SketchySVD(X,r,k);
t0 = sum(cpu);

XX = Q*diag(s(1:r))*P';
err0 = norm(XX-X,'fro')^2/norm(X,'fro')^2;

p1 = 0.1; % choose p1 such that p1*(# of rows) > s
p2 = 0.3; % choose p2 such that p2*(# of columns) > s

[Q,s,P,cpu] = SketchyCoreSVD(X,r,k,p1,p2);
t1 = sum(cpu);

XX = Q*diag(s(1:r))*P';
err1 = norm(XX-X,'fro')^2/norm(X,'fro')^2;