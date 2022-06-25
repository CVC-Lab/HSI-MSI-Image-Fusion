function xx = soft_isotropic(x,tau)

l = length(x)/2;

y = x(1:l);
z = x(l+1:end);

yz = sqrt(y.^2+z.^2+eps);
s = max(yz-tau,0);

y = (y./yz).*s;
z = (z./yz).*s;

xx = [y;z];