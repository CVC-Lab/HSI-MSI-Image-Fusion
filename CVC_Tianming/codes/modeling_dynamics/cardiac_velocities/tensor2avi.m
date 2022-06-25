function tensor2avi(X,FrameRate)
% convert a third-order tensor to grayscale avi

X = (X-min(X(:)))/(max(X(:))-min(X(:)));
TotalFrames = size(X,3);

workingDir = pwd;
outputVideo = VideoWriter(fullfile(workingDir,'Converted.avi'));
outputVideo.FrameRate = FrameRate;
open(outputVideo);

for i = 1:TotalFrames

    im = X(:,:,i);
    clear I
    I(:,:,1) = im;
    I(:,:,2) = im;
    I(:,:,3) = im;
    
    writeVideo(outputVideo,I);
    
end

close(outputVideo);