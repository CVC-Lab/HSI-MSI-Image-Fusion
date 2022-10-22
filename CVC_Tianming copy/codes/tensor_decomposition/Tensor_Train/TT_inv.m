function A = TT_inv(G1,G2,G3)

[n1,r1] = size(G1);
[~,n2,r2] = size(G2);
[~,n3] = size(G3);

A = zeros(n1,n2,n3);

for j = 1:n2
    M = reshape(G2(:,j,:),r1,r2);
    A(:,j,:) = G1*M*G3;
end