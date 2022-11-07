clc; clear; close all

cd ..
cd simulation

run cave.m % generate simulated HSI and MSI
clearvars -except SRI K HSI MSI 

cd ..
cd denoising

[HSI_denoised,~] = denoising(HSI);

cd ..
cd fusion

% Assume we know the true blur kernel is of size no more than 13*13, radius
% is than chosen to be 6
[SRI_fused,K_est,~] = BGLRF_main(HSI_denoised,MSI,10,0,32);

cd ..
cd quality_metrics

% evaluate the fused result
[psnr,rmse,ergas,sam,uiqi,ssim,DD,CCS] = quality_assessment(SRI,SRI_fused,0,1/32);

cd ..
cd fusion