function XX = SketchyCoreTucker(X,r1,r2,r3,k1,k2,k3,p1,p2,p3,q1,q2,q3)

[I1,I2,I3] = size(X);

s1 = 2*k1+1;
s2 = 2*k2+1;
s3 = 2*k3+1;

m1 = floor(p1*I2*I3);
ind1 = randsample(I2*I3,m1);
ind1 = sort(ind1);
X = reshape(X,I1,I2*I3);
T1 = randn(k1,m1);
V1 = X(:,ind1)*T1';

m2 = floor(p2*I3*I1);
ind2 = randsample(I3*I1,m2);
ind2 = sort(ind2);
X = reshape(X,I1,I2,I3);
X = reshape(shiftdim(X,1),I2,[]);
T2 = randn(k2,m2);
V2 = X(:,ind2)*T2';

m3 = floor(p3*I1*I2);
ind3 = randsample(I1*I2,m3);
ind3 = sort(ind3);
X = reshape(X,I2,I3,I1);
X = reshape(shiftdim(X,1),I3,[]);
T3 = randn(k3,m3);
V3 = X(:,ind3)*T3';

m1 = floor(q1*I1);
T1 = randn(s1,m1);
ind1 = randsample(I1,m1);
ind1 = sort(ind1);
m2 = floor(q2*I2);
T2 = randn(s2,m2);
ind2 = randsample(I2,m2);
ind2 = sort(ind2);
m3 = floor(q3*I3);
T3 = randn(s3,m3);
ind3 = randsample(I3,m3);
ind3 = sort(ind3);

H = T3*X(ind3,:);
H = reshape(H,s3,I1,I2);
H = shiftdim(H,1);
H = reshape(H,I1,I2*s3);
H = T1*H(ind1,:);
H = reshape(H,s1,I2,s3);
H = shiftdim(H,1);
H = reshape(H,I2,s3*s1);
H = T2*H(ind2,:);
H = reshape(H,s2,s3,s1);
H = shiftdim(H,2);

[Q1,~] = qr(V1,0);
[Q2,~] = qr(V2,0);
[Q3,~] = qr(V3,0);

P1 = T1*Q1(ind1,:);
P1 = pinv(P1);
P2 = T2*Q2(ind2,:);
P2 = pinv(P2);
P3 = T3*Q3(ind3,:);
P3 = pinv(P3);

W = ttm(tensor(H),{P1,P2,P3},[1 2 3]); % k1*k2*k3

[G,U] = HOOI(W,[r1 r2 r3]);

U1 = U{1};
U2 = U{2};
U3 = U{3};

P1 = Q1*U1;
P2 = Q2*U2;
P3 = Q3*U3;

P{1} = P1;
P{2} = P2;
P{3} = P3;

XX = ttensor(G,P);