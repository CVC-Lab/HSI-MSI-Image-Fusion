function [Q,s,P,cpu] = SketchyCoreSVD(A,r,k,p1,p2)
% SketchyCoreSVD: SketchySVD from Random Subsampling of the Data Matrix

[M,N] = size(A);

s = 2*k+1;

cpu = zeros(1,8);

m = floor(p1*M);
n = floor(p2*N);

% form X
tic;
row = randsample(M,m);
row = sort(row);
T1 = randn(k,m);
X = T1*A(row,:);
clear T1
cpu(1) = toc;

% form Y
tic;
col = randsample(N,n);
col = sort(col);
T2 = randn(k,n);
Y = A(:,col)*T2';
clear T2
cpu(2) = toc;

% form Z
tic;
T3 = randn(s,m);
T4 = randn(s,n);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % to get independence (in the proof)
%row = randsample(M,m);
%row = sort(row);
%col = randsample(N,n);
%col = sort(col);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if M < N
    Z = T3*(A(row,col)*T4');
else
    Z = (T3*A(row,col))*T4';
end
cpu(3) = toc;

% QR of X
tic;
[P,~] = qr(X',0);
cpu(4) = toc;

% QR of Y
tic;
[Q,~] = qr(Y,0);
cpu(5) = toc;

% form C
tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%L = pinv(T3*Q(row,:));
%R = pinv(T4'*P(col,:));
%C = L*Z*R';
L = T3*Q(row,:);
clear T3
R = T4*P(col,:);
clear T4
C = (L\Z)/R';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cpu(6) = toc;

% SVD of C
tic;
[U,S,V] = svd(C);
s = diag(S);
cpu(7) = toc;

% final
tic;
%Q = Q*U;
Q = Q*U(:,1:r);
%P = P*V;
P = P*V(:,1:r);
cpu(8) = toc;