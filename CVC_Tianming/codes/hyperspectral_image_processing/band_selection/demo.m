clc; clear; close all

cd ..
cd test_data

load('indian_pines.mat')

cd ..
cd denoising

%X = indian_pines_corrected/scaling;
X = indian_pines_corrected;
[Y,~] = denoising(X);

n3 = size(X,3);
X = reshape(X,[],n3);
Y = reshape(Y,[],n3);

cd ..
cd band_selection

% select 16 most informative bands
IND = MNBS(X,X-Y,16); % get indices for the selected bands 

Z = Y(:,IND); % denosied HSI with selected bands for further processing