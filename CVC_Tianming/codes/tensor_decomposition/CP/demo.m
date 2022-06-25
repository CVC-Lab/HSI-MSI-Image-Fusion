clc; clear; close all

cd ..
cd test_data

load('Cardiac_DE_2.mat')

cd ..
cd tensor_toolbox

r = 20; % cp rank

XX = cp_als(tensor(X),r);
XX = double(XX);

err = norm(XX(:)-X(:))^2/norm(X(:))^2;

cd ..
cd CP