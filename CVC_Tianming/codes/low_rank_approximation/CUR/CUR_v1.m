function [C,U,R] = CUR_v1(X,r,m,n)
% CUR decomposition based on randomly sampled columns and rows
%
% ``CUR decompositions, approximations, and perturbations''

[M,N] = size(X);

id_col = randsample(N,n);
id_col = sort(id_col);
id_col = id_col';

id_row = randsample(M,m);
id_row = sort(id_row);

C = X(:,id_col);

R = X(id_row,:);

A = X(id_row,id_col);
[Q,S,P] = svds(A,r);
s = diag(S);
U = bsxfun(@times,P,(1./s)');
U = U*Q';