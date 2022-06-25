clc; clear; close all

cd ..
cd test_data

load('indian_pines.mat')

cd ..
cd denoising

X = indian_pines_corrected/scaling;
[Y,SNR_dB] = denoising(X);

plot(SNR_dB); % estimated SNR of each band

% check the effects of denoising
I1 = X(:,:,2);
I1 =(I1-min(I1(:)))/(max(I1(:))-min(I1(:)));
I2 = Y(:,:,2);
I2 =(I2-min(I2(:)))/(max(I2(:))-min(I2(:)));
imtool([I1 I2])