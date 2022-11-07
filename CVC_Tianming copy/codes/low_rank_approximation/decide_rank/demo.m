clc; clear; close all

% if full SVD can be computed, one can decide the proper rank of a dataset
% based on scree plot

cd ..
cd test_data

load('YaleFace.mat')

cd ..
cd decide_rank

[~,S,~] = svd(X,'econ');
s = diag(S);

v = scree_plot(s);

% one heuristic is to choose the rank at the elbow of the curve
plot(1:size(v),v);
xlabel('rank');title('scree plot');grid on;