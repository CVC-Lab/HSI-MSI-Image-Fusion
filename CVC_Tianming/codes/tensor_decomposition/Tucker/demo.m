clc; clear; close all

cd ..
cd test_data

load('Cardiac_DE_2.mat')
norm_X = norm(X(:))^2;

cd ..
addpath('tensor_toolbox')
cd Tucker

r1 = 20; r2 = 20; r3 = 5; % multilinear ranks

tic;
[core,U] = HOOI(tensor(X),[r1 r2 r3]); % HOOI
XX = ttensor(core,U);
t0 = toc;

norm_XX = norm(XX)^2;
inner = innerprod(tensor(X),XX);
err0 = (norm_X+norm_XX-2*inner)/norm_X;

k1 = 2*r1+1;
k2 = 2*r2+1;
k3 = 4*r3+1;

tic;
XX = SketchyTucker(X,r1,r2,r3,k1,k2,k3);
t1 = toc;

norm_XX = norm(XX)^2;
inner = innerprod(tensor(X),XX);
err1 = (norm_X+norm_XX-2*inner)/norm_X;

p1 = 0.03; % p1*N2N3 > s1  
p2 = 0.03; % p2*N1N3 > s2
p3 = 0.03; % p3*N1N2 > s3
q1 = 0.6; % q1*N1 > s1
q2 = 0.6; % q2*N2 > s2
q3 = 0.3; % q3*N3 > s3

tic;
XX = SketchyCoreTucker(X,r1,r2,r3,k1,k2,k3,p1,p2,p3,q1,q2,q3);
t2 = toc;

norm_XX = norm(XX)^2;
inner = innerprod(tensor(X),XX);
err2 = (norm_X+norm_XX-2*inner)/norm_X;

cd ..
rmpath('tensor_toolbox')
cd Tucker