clc; clear; close all

cd ..
cd matfiles

load('cave_fake_and_real_beers_ms.mat')

cd ..
cd denoising

% HSI = lr_hsi;
MSI = hr_msi;
MSI = MSI/255;
SRI = hr_hsi;
HSI = lr_hsi;

scaling = max(SRI, [], 'all');

SRI = SRI/scaling;
HSI = HSI/scaling;
% [SRI,~] = denoising(SRI);
% N1 = size(MSI,1)
% N2 = size(MSI,2)
% N3 = size(MSI,3)
% 
% SRI = SRI(1:N1,1:N2,:);
% 
% cd ..
% cd simulation
% 
% % get the wavelength of each band
% wave = 400:(2500-400)/(220-1):2500
% ind = setdiff(1:220,bands_removed);
% wave = wave(ind);

% generate MSI using spectral responses from the sensor of Landsat_TM5 
% satellite, to simulate another sensor, need to get its spectral responses

% load('Landsat_TM5.mat')
% S{1} = blue;
% S{2} = green;
% S{3} = red;
% S{4} = nir;
% S{5} = swir1;
% S{6} = swir2;
% 
% n3 = 6;
% P3 = zeros(N3,n3);
% 
% for i = 1:n3
%     
%     s = S{i};
%     
%     temp = s(:,1);
%     TEMP = s(:,2);
%   
%     IND1 = find(wave>temp(1));
%     ind1 = IND1(1);
%     IND2 = find(wave<temp(end));
%     ind2 = IND2(end);
%     
%     yy = spline(temp,TEMP,wave(ind1:ind2));
%     
%     P3(ind1:ind2,i) = yy;
%     P3(:,i) = P3(:,i)/sum(P3(:,i));
%     
% end
% 
% SRI = reshape(SRI,[],N3);
% MSI = SRI*P3;
% MSI = reshape(MSI,N1,N2,n3);
% SRI = reshape(SRI,N1,N2,N3);

% add noise to the MSI
% [MSI,noise_tenM] = add_noise(MSI,40); % noisy MSI of SNR about 40

% generate HSI of downsampling ratio 4, with periodic boundary condition
% ratio = 32;
% K0 = fspecial('gaussian',[9 9],ratio/(2.355)); % 9 = 2*4+1

% blur kernel shifted 2 pixels vertically and horizontally to the top left
% radius = 4;
% K = zeros(2*radius+1,2*radius+1);
% K(1:9,1:9) = K0;
% 
% center = zeros(1,2);
% center(1) = N1/2+1;
% center(2) = N2/2+1;
% 
% I = (center(1)-radius):(center(1)+radius);
% I = I.';
% I = repmat(I,[1 2*radius+1]);
% I = I(:);
% J = (center(2)-radius):(center(2)+radius);
% J = repmat(J,[2*radius+1 1]);
% J = J(:);
% ind = sub2ind([N1 N2],I,J); % indices for blur kernel
% 
% KK = zeros(N1,N2);
% KK(ind) = K;
% Fk = fft2(circshift(KK,1-center));
% 
% n1 = 16;
% n2 = 16;
% HSI = zeros(n1,n2,N3);
% 
% for band = 1:N3
%     
%     x = SRI(:,:,band); 
%     Fx = fft2(x);
%     x = real(ifft2(Fk.*Fx));
%     HSI(:,:,band) = x(2:ratio:end,2:ratio:end);
%     
% end
% 
% % add noise to the HSI
% [HSI,noise_tenH] = add_noise(HSI,35); % noisy HSI of SNR about 35