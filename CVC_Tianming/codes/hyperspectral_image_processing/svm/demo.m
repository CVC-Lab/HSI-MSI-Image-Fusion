clc; clear; close all

cd ..
cd test_data

load('indian_pines.mat')

X = indian_pines_corrected/scaling;
n3 = size(X,3);
X = reshape(X,[],n3);
Y = indian_pines_gt;
Y = Y(:);
Lbls = [2 3 5 6 8 10 11 12 14]; % 9 out of 16 classes are chosen

cd ..
cd simulation

run demo.m

cd ..
cd fusion

run demo.m

n3 = size(SRI,3);

SRIfused_SVM = reshape(SRI_fused,[],n3);
SRI_SVM = reshape(SRI,[],n3);

cd ..
cd test_data

load('indian_pines.mat')
Y = indian_pines_gt;
Y = Y(:);

cd ..
cd svm

Lbls = [2 3 5 6 8 10 11 12 14]; % 9 out of 16 classes are chosen

[OA,OA_overall] = my_svm(SRIfused_SVM,Y,Lbls,0.1); % train on 10 percent samples
close all;

oa = mean(OA,2); % classification accuracy of each class
oa_overall = mean(OA_overall); % overall classification accuracy