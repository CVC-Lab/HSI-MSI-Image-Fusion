import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Define the upsampling layers
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.upconv1 = nn.ConvTranspose2d(64, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        # BatchNorm layers for normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

    def forward(self, hsi, msi):
        # Encoder
        x1 = F.relu(self.bn1(self.conv1(msi)))
        x2 = F.relu(self.bn2(self.conv2(F.max_pool2d(x1, 2))))
        x3 = F.relu(self.bn3(self.conv3(F.max_pool2d(x2, 2))))
        x4 = F.relu(self.bn4(self.conv4(F.max_pool2d(x3, 2))))
        # Decoder
        x = F.relu(self.upconv4(x4))
        x = F.relu(self.upconv3(x))
        x = self.upconv2(x)
        return x

if __name__ == '__main__':
    # Test the model
    model = CNN(3, 5)  # Assume input channels = 3 and output segmentation map has 5 classes
    for i in range(1, 5):
        input_image = torch.rand(2, 3, 256*i, 256*i)
        output = model(input_image)
        print(output.shape)
