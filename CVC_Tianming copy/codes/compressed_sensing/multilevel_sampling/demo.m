% The demo below is for wavelets
clc;clear
load('Shepp-Logan.mat') % test image, 1024 by 1024
addpath('../spgl1-1.9')

I = I_512;

%% Step 1: through trail and error, find a sparse approximation of the image in the wavelet domain
level = 5;
per = 0.03;
[II,S] = sparse_approx(I,level,per);
PSNR21 = psnr(II,I); % model error

%% Step 2: perform sampling
[n1,n2] = size(I);
N = n1*n2;
m = floor(per*N);
const = 6;
M = const*m;

c = 2; % 1 - uniform, 2 - Fourier+wavelet, 3 - Hadamard+wavelet

switch c
    case 1
        % uniform sampling
        IND = randsample(n1*n2,M); % uniform sampling
        IND = sort(IND);
    case 2
        % multi-level sampling (Fourier+wavelet)
        nu = round(log2(min(n1,n2)));
        bounds = [2^(nu-4),2^(nu-2),2^(nu-1)];
        pos = random_circ_subsamp(n1,n2,bounds,M);
        IND = sub2ind([n1,n2],pos(:,1),pos(:,2));
        IND = sort(IND);
    case 3
        % multi-level sampling (Hadamard+wavelet)
        scheme = uniform_rect_samp_scheme(n1,n2,M);
        pos = random_rect_subsamp(scheme);
        IND = sub2ind([n1,n2],pos(:,1),pos(:,2));
        IND = sort(IND);
end

%% Step 3: recovery from samples

switch c
    case 1
        b = I(IND);
        bb = II(IND);
        sigma = norm(II(IND)-I(IND));
        opA = @(x,mode) partial_Uniform(x,S,IND,n1,n2,level,mode);
        C = spg_bpdn(opA,b,sigma);
        III = waverec2(C,S,'haar');
        III = reshape(III,n1,n2);
    case 2
        
        % SPGL1 for Fourier
        %b = fftshift(fft2(I))/sqrt(n1*n2);
        %b = b(IND);
        %bb = fftshift(fft2(II))/sqrt(n1*n2);
        %bb = bb(IND);
        %sigma = norm(bb-b);
        %opA = @(x,mode) partial_Fourier(x,S,IND,n1,n2,level,mode);
        
        % IHT for Fourier
        b = fftshift(fft2(I));
        b = b(IND)/sqrt(n1*n2);
        III = DIC_IHT_Fourier(b,IND,M,n1,n2,level);
        
    case 3
        b = fwht2(I);
        b = b(IND)*sqrt(n1*n2);
        bb = fwht2(II);
        bb = bb(IND)*sqrt(n1*n2);
        sigma = norm(bb-b);
        opA = @(x,mode) partial_Hadamard(x,S,IND,n1,n2,level,mode);
        C = spg_bpdn(opA,b,sigma);
        III = waverec2(C,S,'db4');
        III = reshape(III,n1,n2);
end

PSNR32 = psnr(III,II); % recovery error
PSNR31 = psnr(III,I);  

rmpath('../spgl1-1.9')