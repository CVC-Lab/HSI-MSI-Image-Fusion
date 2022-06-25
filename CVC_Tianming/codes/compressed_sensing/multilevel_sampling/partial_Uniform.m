function y = partial_Uniform(x,S,IND,n1,n2,level,mode)

if mode == 1
    x = reshape(x,1,[]);
    z = waverec2(x,S,'haar');
    y = z(IND);
else
    x = reshape(x,[],1);
    z = zeros(n1*n2,1);
    z(IND) = x;
    z = reshape(z,n1,n2);
    [y,~] = wavedec2(z,level,'haar');
    y = reshape(y,[],1);
end