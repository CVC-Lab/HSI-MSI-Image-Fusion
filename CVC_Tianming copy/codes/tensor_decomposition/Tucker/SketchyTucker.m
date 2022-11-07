function XX = SketchyTucker(X,r1,r2,r3,k1,k2,k3)
% ``Low-rank Tucker approximation of a tensor from streaming data''
    
[I1,I2,I3] = size(X);

s1 = 2*k1+1;
s2 = 2*k2+1;
s3 = 2*k3+1;

X = reshape(X,I1,I2*I3);
T1 = randn(k1,I2*I3);
V1 = X*T1';

X = reshape(X,I1,I2,I3);
X = reshape(shiftdim(X,1),I2,[]);
T2 = randn(k2,I3*I1);
V2 = X*T2';

X = reshape(X,I2,I3,I1);
X = reshape(shiftdim(X,1),I3,[]);
T3 = randn(k3,I1*I2);
V3 = X*T3';

T1 = randn(s1,I1);
T2 = randn(s2,I2);
T3 = randn(s3,I3);

H = T3*X;
H = reshape(H,s3,I1,I2);
H = shiftdim(H,1);
H = reshape(H,I1,I2*s3);
H = T1*H;
H = reshape(H,s1,I2,s3);
H = shiftdim(H,1);
H = reshape(H,I2,s3*s1);
H = T2*H;
H = reshape(H,s2,s3,s1);
H = shiftdim(H,2);

[Q1,~] = qr(V1,0);
[Q2,~] = qr(V2,0);
[Q3,~] = qr(V3,0);

P1 = T1*Q1;
P1 = pinv(P1);
P2 = T2*Q2;
P2 = pinv(P2);
P3 = T3*Q3;
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