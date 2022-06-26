clc; clear; close all

cd ..
cd simulation

run demo.m % generate simulated HSI and MSI
clearvars -except SRI K HSI MSI 

cd ..
cd denoising

[HSI_denoised,~] = denoising(HSI);

cd ..
cd fusion

% Assume we know the true blur kernel is of size no more than 13*13, radius
% is than chosen to be 6
[SRI_fused,K_est,~] = BGLRF_main(HSI_denoised,MSI,10,10,6);

cd ..
cd quality_metrics

% evaluate the fused result
[psnr,rmse,ergas,sam,uiqi,ssim,DD,CCS] = quality_assessment(SRI,SRI_fused,0,1/4)

cd ..
cd fusion

% check the accuracy visually
I_SRI = SRI(:,:,2);
I_SRI =(I_SRI-min(I_SRI(:)))/(max(I_SRI(:))-min(I_SRI(:)));
I_SRIfused = SRI_fused(:,:,2);
I_SRIfused =(I_SRIfused-min(I_SRIfused(:)))/(max(I_SRIfused(:))-min(I_SRIfused(:)));
imtool([I_SRI I_SRIfused])