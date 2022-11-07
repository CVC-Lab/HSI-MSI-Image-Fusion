from .dbglrf import AutoEncoder, calc_loss
from datasets.cave_dataset import CAVEDataset
from torch.optim import Adam
import torch
import pdb

epochs = 10
batch_size = 8
in_channels = 31
dec_channels = 31
lr = 1e-3

train_dataset = CAVEDataset("./datasets/data/CAVE", None, mode="train")
test_dataset = CAVEDataset("./datasets/data/CAVE", None, mode="test")
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          shuffle=False, batch_size=batch_size)

model = AutoEncoder(in_channels=in_channels, dec_channels=dec_channels)
optimizer = Adam(model.parameters(), lr=lr)
#lr_hsi[31, 64, 64], hrmsi[31, 512, 512], hr_hsi[3, 512, 512]
# x = torch.rand(31, 512, 512)
# y = torch.rand(31, 64, 64)
# z = torch.rand(512, 512, 3).numpy()

model.train()
for epoch in range(epochs):
    train_recon_loss = 0
    train_gl_loss = 0
    for items in train_loader:
        optimizer.zero_grad()
        c, y, z, x, lz = items
        y_hat, x_hat = model(x)
        recon_loss, GL = calc_loss(x_hat, y_hat, lz, y)
        total_loss = recon_loss.sum() + GL.sum()
        total_loss.backward()
        optimizer.step()
        train_recon_loss += recon_loss.sum().item()
        train_gl_loss += GL.sum().item()

    if epoch % 2 == 0:
        print(f"epoch: {epoch} \
        train reconstruction loss: {train_recon_loss/len(train_loader)}, \
            GLaplacian loss: {train_gl_loss/len(train_loader)}")
    
