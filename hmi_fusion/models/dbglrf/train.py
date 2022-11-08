from .dbglrf import AutoEncoder, calc_loss
from datasets.cave_dataset import CAVEDataset
import numpy as np
from torch.optim import Adam
import torch
import pdb

epochs = 40
batch_size = 8
in_channels = 31
dec_channels = 31
lr = 1e-3
load_best_model = True
model_path = "./artifacts/best_model.pth"

train_dataset = CAVEDataset("./datasets/data/CAVE", None, mode="train")
test_dataset = CAVEDataset("./datasets/data/CAVE", None, mode="test")
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          shuffle=False, batch_size=batch_size)

model = AutoEncoder(in_channels=in_channels, dec_channels=dec_channels)
if load_best_model:
    model.load_state_dict(torch.load(model_path))

optimizer = Adam(model.parameters(), lr=lr)

print("starting training")

alpha, beta, gamma, delta = 1, 1, 1.5, 1
best_model_test_loss = np.inf
for epoch in range(epochs):
    model.train()
    train_recon_loss_x = 0
    train_recon_loss_y = 0
    train_sam_loss = 0
    train_gl_loss = 0
    for items in train_loader:
        optimizer.zero_grad()
        c, y, z, x, lz = items
        y_hat, x_hat = model(x)
        recon_loss_x, sam_loss, recon_loss_y, GL = calc_loss(x_hat, y_hat, lz, x, y)
        total_loss = alpha*recon_loss_x + beta*recon_loss_y + gamma*sam_loss + delta*GL 
        total_loss.backward()
        optimizer.step()
        train_recon_loss_x += recon_loss_x.item()
        train_recon_loss_y += recon_loss_y.item()
        train_sam_loss += sam_loss.item() 
        train_gl_loss += GL.item()
    # get eval loss every 2 epochs
    if epoch % 2 == 0:
        print(f"train epoch: {epoch} \
        recon_x loss: {train_recon_loss_x/len(train_loader)}, \
        recon_y loss: {train_recon_loss_y/len(train_loader)}, \
        sam_loss loss: {train_sam_loss/len(train_loader)}, \
        GLaplacian loss: {train_gl_loss/len(train_loader)}")
        model.eval()
        test_recon_loss_x = 0
        test_recon_loss_y = 0
        test_sam_loss = 0
        test_gl_loss = 0
        total_test_loss = 0
        for items in test_loader:
            c, y, z, x, lz = items
            y_hat, x_hat = model(x)
            recon_loss_x, sam_loss, recon_loss_y, GL = calc_loss(x_hat, y_hat, lz, x, y)
            total_loss = recon_loss_x + recon_loss_y + sam_loss + GL 
            test_recon_loss_x += recon_loss_x.item()
            test_recon_loss_y += recon_loss_y.item()
            test_sam_loss += sam_loss.item() 
            test_gl_loss += GL.item()
            total_test_loss = total_loss.item()

        print(f"test epoch: {epoch} \
        recon_x loss: {test_recon_loss_x/len(test_loader)}, \
        recon_y loss: {test_recon_loss_y/len(test_loader)}, \
        sam_loss loss: {test_sam_loss/len(test_loader)}, \
        GLaplacian loss: {test_gl_loss/len(test_loader)}")
        total_test_loss /= len(test_loader)
    
        if best_model_test_loss > total_test_loss:
           best_model_test_loss =  best_model_test_loss
           print("saving ...")
           torch.save(model.state_dict(), model_path)

    

