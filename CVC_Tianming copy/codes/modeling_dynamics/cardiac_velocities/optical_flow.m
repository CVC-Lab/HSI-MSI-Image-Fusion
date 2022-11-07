function [Vx,Vy] = optical_flow(Name)
% calculate velocity fields from the input video using optical flow

vidReader = VideoReader(Name);

opticFlow = opticalFlowHS; % Horn-Schunck method

k = 0;
while hasFrame(vidReader)
    frameRGB = readFrame(vidReader);
    frameGray = rgb2gray(frameRGB);
    flow = estimateFlow(opticFlow,frameGray);
    if k > 0
        Vx(:,:,k) = double(flow.Vx);
        Vy(:,:,k) = double(flow.Vy);
        %O(:,:,k) = double(flow.Orientation);
        %M(:,:,k) = double(flow.Magnitude);
    end
    k = k+1;   
end