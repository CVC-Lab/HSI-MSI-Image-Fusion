function X = fwht2(X)

n = size(X,1);

X = fwht(X);
X = X';
X = fwht(X);
X = X';
X = n*X;