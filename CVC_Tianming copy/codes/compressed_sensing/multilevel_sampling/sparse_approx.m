function [I,S] = sparse_approx(I,level,per)

I = double(I);
[n1,n2] = size(I);

[C,S] = wavedec2(I,level,'haar');

% N = zeros(1,level+4);
% N(1) = 0;
% N(2) = S(2,1)*S(2,2);
% N(3) = 2*S(2,1)*S(2,2);
% N(4) = 3*S(2,1)*S(2,2);
% N(5) = 4*S(2,1)*S(2,2);
% for i = 6:level+4
%     N(i) = N(i-1)+3*S(i-3,1)*S(i-3,2);
% end
% 
% s = zeros(1,level+3);
% A = appcoef2(C,S,'haar',level); 
% s(1) = sparsity(A(:)); 
% [H,V,D] = detcoef2('all',C,S,level);
% s(2) = sparsity(H(:));
% s(3) = sparsity(V(:));
% s(4) = sparsity(D(:));
% for l = level-1:-1:1
%     d = detcoef2('compact',C,S,l);
%     s(level+4-l) = sparsity(d);
% end
% 
% for i = 1:level+3
%     ind = N(i)+1:N(i+1);
%     c = C(ind);
%     [~,I] = sort(abs(c),'descend');
%     C(ind) = 0;
%     I = I(1:s(i));
%     C(ind(I)) = c(I);
% end

m = floor(per*n1*n2);

[~,ind] = sort(abs(C),'descend');
CC = C;
C = zeros(size(CC));
C(ind(1:m)) = CC(ind(1:m));

I = waverec2(C,S,'haar');