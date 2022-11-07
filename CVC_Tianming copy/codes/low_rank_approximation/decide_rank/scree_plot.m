function v = scree_plot(s)

n = length(s);

v = zeros(n-1,1);

for r = 1:n-1
    v(r) = sum(s(r+1:end).^2)/sum(s.^2);
end