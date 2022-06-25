function [d,ind,X] = SSRFT(A,k)

M = size(A,1);
d = rand(M,1);
d = d > 0.5;
d = 2*d-1;
TEMP1 = bsxfun(@times,d,A);
TEMP2 = dct(TEMP1);
clear TEMP1
ind = randsample(M,k);
ind = sort(ind);
X = TEMP2(ind,:);
clear TEMP2