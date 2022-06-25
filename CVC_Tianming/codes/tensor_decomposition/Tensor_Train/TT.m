function [G1,G2,G3] = TT(A,r1,r2)

[n1,n2,n3] = size(A);

C = reshape(A,n1,n2*n3);
clear A
[G1,S,V] = svds(C,r1);
s = diag(S);
C = bsxfun(@times,s,V');

C = reshape(C,r1*n2,n3);
[G2,S,V] = svds(C,r2);
s = diag(S);
G2 = reshape(G2,r1,n2,r2);
G3 = bsxfun(@times,s,V');