function [M,noise_tenM] = add_noise(M,SNR)

[N1,N2,N3] = size(M);

signal_power = norm(M(:))^2/(N1*N2*N3);
noise_power = signal_power/10^(SNR/10);
noise_tenM = sqrt(noise_power)*randn(N1,N2,N3);

M = M+noise_tenM;