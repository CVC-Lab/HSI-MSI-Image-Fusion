% part1 is about computing velocity fields from a third-order tensor
clc; clear; close all

cd ..
cd test_data

load('Cardiac_CD_23.mat')

cd ..
cd cardiac_velocities

FrameRate = 10; % 10 frames per second
tensor2avi(X,FrameRate) % generate a video named 'Converted.avi'

[Vx,Vy] = optical_flow('Converted.avi');

clearvars -except Vx Vy
save('Velocities.mat')
delete Converted.avi