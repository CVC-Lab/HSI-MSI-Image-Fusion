function [C,U,R] = CUR_v2(X,r,m,n)
% CUR decomposition based on column/row lengths 
%
% ``CUR decompositions, approximations, and perturbations''

% sampling based on column/row lengths
prob_col = sum(X.^2,1);
prob_row = sum(X.^2,2);
prob_col = prob_col/sum(prob_col);
prob_row = prob_row/sum(prob_row);
prob_col_cum = cumsum(prob_col);
prob_row_cum = cumsum(prob_row);

id_col = zeros(1,n);
id_row = zeros(m,1);

for j = 1:n
    p = rand(1);
    id = find(prob_col_cum>p);
    id_col(j) = id(1);
end
id_col = sort(id_col);

C = X(:,id_col);

for i = 1:m
    p = rand(1);
    id = find(prob_row_cum>p);
    id_row(i) = id(1);
end
id_row = sort(id_row);

R = X(id_row,:);

A = X(id_row,id_col);
[Q,S,P] = svds(A,r);
s = diag(S);
U = bsxfun(@times,P,(1./s)');
U = U*Q';