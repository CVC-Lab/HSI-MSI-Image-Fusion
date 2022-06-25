function T = SSM(k,M)

if k < 9
    error('k is too small')
end

p = 8; % change here

N = p*M;

ind1 = zeros(N,1);
for j = 1:M
    idx = (j-1)*p+1:j*p;
    ind1(idx) = randperm(k,p);
end

%ind1 = rand(p,M);
%ind1 = ceil(k*ind1);
%ind1 = ind1(:);

ind2 = 1:M;
ind2 = repmat(ind2,p,1);
ind2 = ind2(:);

v = sign(randn(N,1));
T = sparse(ind1,ind2,v);