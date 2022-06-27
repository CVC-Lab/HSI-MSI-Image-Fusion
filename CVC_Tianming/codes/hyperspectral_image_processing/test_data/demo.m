%Load original image
% load('indian_pines.mat')
% 
% X = indian_pines_corrected/scaling;
% I1 = X(:,:,2);
% I1 =(I1-min(I1(:)))/(max(I1(:))-min(I1(:)));
% imtool(I1)


% Display HSI used for fusion
I1 = HSI(:,:,2);
I1 =(I1-min(I1(:)))/(max(I1(:))-min(I1(:)));

I2 = HSI(:,:,10);
I2 =(I2-min(I2(:)))/(max(I2(:))-min(I2(:)));

I3 = HSI(:,:,18);
I3 =(I3-min(I3(:)))/(max(I3(:))-min(I3(:)));

I4 = HSI(:,:,26);
I4 =(I4-min(I4(:)))/(max(I4(:))-min(I4(:)));

I5 = HSI(:,:,34);
I5 =(I5-min(I5(:)))/(max(I5(:))-min(I5(:)));

I6 = HSI(:,:,42);
I6 =(I6-min(I6(:)))/(max(I6(:))-min(I6(:)));

I7 = HSI(:,:,50);
I7 =(I7-min(I7(:)))/(max(I7(:))-min(I7(:)));

I8 = HSI(:,:,58);
I8 =(I8-min(I8(:)))/(max(I8(:))-min(I8(:)));

I9 = HSI(:,:,66);
I9 =(I9-min(I9(:)))/(max(I9(:))-min(I9(:)));

I10 = HSI(:,:,74);
I10 =(I10-min(I10(:)))/(max(I10(:))-min(I10(:)));

I11 = HSI(:,:,82);
I11 =(I11-min(I11(:)))/(max(I11(:))-min(I11(:)));

I12 = HSI(:,:,90);
I12 =(I12-min(I12(:)))/(max(I12(:))-min(I12(:)));

I13 = HSI(:,:,98);
I13 =(I13-min(I13(:)))/(max(I13(:))-min(I13(:)));

I14 = HSI(:,:,106);
I14 =(I14-min(I14(:)))/(max(I14(:))-min(I14(:)));

I15 = HSI(:,:,114);
I15 =(I15-min(I15(:)))/(max(I15(:))-min(I15(:)));

I16 = HSI(:,:,122);
I16 =(I16-min(I16(:)))/(max(I16(:))-min(I16(:)));

I17 = HSI(:,:,130);
I17 =(I17-min(I17(:)))/(max(I17(:))-min(I17(:)));

I18 = HSI(:,:,138);
I18 =(I18-min(I18(:)))/(max(I18(:))-min(I18(:)));

I19 = HSI(:,:,146);
I19 =(I19-min(I19(:)))/(max(I19(:))-min(I19(:)));

I20 = HSI(:,:,154);
I20 =(I20-min(I20(:)))/(max(I20(:))-min(I20(:)));

I21 = HSI(:,:,162);
I21 =(I21-min(I21(:)))/(max(I21(:))-min(I21(:)));

I22 = HSI(:,:,170);
I22 =(I22-min(I22(:)))/(max(I22(:))-min(I22(:)));

I23 = HSI(:,:,178);
I23 =(I23-min(I23(:)))/(max(I23(:))-min(I23(:)));

I24 = HSI(:,:,186);
I24 =(I24-min(I24(:)))/(max(I24(:))-min(I24(:)));

imtool([I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,I14,I15,I16,I17,I18,I19,I20,I21,I22,I23,I24])


%Display MSI
% I1 = MSI(:,:,1);
% I1 =(I1-min(I1(:)))/(max(I1(:))-min(I1(:)));
% 
% I2 = MSI(:,:,2);
% I2 =(I2-min(I2(:)))/(max(I2(:))-min(I2(:)));
% 
% I3 = MSI(:,:,3);
% I3 =(I3-min(I3(:)))/(max(I3(:))-min(I3(:)));
% 
% I4 = MSI(:,:,4);
% I4 =(I4-min(I4(:)))/(max(I4(:))-min(I4(:)));
% 
% I5 = MSI(:,:,5);
% I5 =(I5-min(I5(:)))/(max(I5(:))-min(I5(:)));
% 
% I6 = MSI(:,:,6);
% I6 =(I6-min(I6(:)))/(max(I6(:))-min(I6(:)));
% 
% imtool([I1,I2,I3,I4,I5,I6])
% 
