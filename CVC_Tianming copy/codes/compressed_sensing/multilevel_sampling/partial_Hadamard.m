function y = partial_Hadamard(x,S,IND,n1,n2,level,mode)

if mode == 1
    x = reshape(x,1,[]);
    z = waverec2(x,S,'haar');
    y = fwht2(z)*sqrt(n1*n2);
    y = y(IND);
else
    x = reshape(x,[],1);
    z = zeros(n1,n2);
    z(IND) = x;
    z = fwht2(z)*sqrt(n1*n2);
    [y,~] = wavedec2(z,level,'haar');
    y = reshape(y,[],1);
end