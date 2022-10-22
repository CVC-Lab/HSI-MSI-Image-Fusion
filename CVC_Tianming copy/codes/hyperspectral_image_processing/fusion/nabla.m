function D = nabla(p,q)
% calculate the gradient in x and y directions, by defining a (sparse) matrix

l = p*q;
Dx = zeros(l,l);
Dy = zeros(l,l);

for i = 1:p
    for j = 1:q
        ind1 = (j-1)*p+i;
        Dx(ind1,ind1) = 1;
        if j > 1
            ind2 = ind1-p;
            Dx(ind1,ind2) = -1;
        end
    end
end

for i = 1:p
    for j = 1:q
        ind1 = (j-1)*p+i;
        Dy(ind1,ind1) = 1;
        if i > 1 
            ind2 = ind1-1;
            Dy(ind1,ind2) = -1;
        end
    end
end

D = [Dx;Dy];
D = sparse(D);