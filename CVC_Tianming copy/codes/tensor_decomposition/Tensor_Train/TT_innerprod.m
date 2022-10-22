function inner = TT_innerprod(G1,G2,G3,A)

[n1,r1] = size(G1);
[~,n2,r2] = size(G2);
[~,n3] = size(G3);

inner = 0;

for j = 1:n2
    M = reshape(G2(:,j,:),r1,r2);
    M = G1*M*G3;
    MM = reshape(A(:,j,:),n1,n3);
    inner = inner+sum(sum(M.*MM));
end