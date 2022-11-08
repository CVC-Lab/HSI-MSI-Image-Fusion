from .dbglrf import AutoEncoder2
from datasets.cave_dataset import CAVEDataset
import numpy as np
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import torch
import os
import pdb

## config
epochs = 40
batch_size = 8
y_channel = 31
z_channel = 3
lr = 1e-3
model_name = "ae2"
load_best_model = False
model_path = f"./artifacts/{model_name}/gmodel.pth"
if not os.path.exists(f"./artifacts/{model_name}"):
    os.makedirs(f"./artifacts/{model_name}")

writer = SummaryWriter("./artifacts/{model_name}")
train_dataset = CAVEDataset("./datasets/data/CAVE", None, mode="train")
test_dataset = CAVEDataset("./datasets/data/CAVE", None, mode="test")
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          shuffle=False, batch_size=batch_size)
model = AutoEncoder2(y_channel, z_channel)
if load_best_model:
    model.load_state_dict(torch.load(model_path))
optimizer = Adam(model.parameters(), lr=lr)
print("starting training")
alpha, beta, gamma, delta = 1, 1, 1.5, 1
best_model_test_loss = np.inf
for epoch in range(epochs):
    model.train()
    train_recon_loss_x1 = 0
    train_recon_loss_x2 = 0
    train_sam_loss = 0
    train_gl_loss = 0
    total_train_loss = 0
    for items in train_loader:
        optimizer.zero_grad()
        c, y, z, x, lz = items
        
        x1, x2 = model(y, z)
        recon_loss_x1, recon_loss_x2, sam_loss, GL = model.calc_loss(x1, x2, lz, x)
        total_loss = alpha*recon_loss_x1 + beta*recon_loss_x2 + gamma*sam_loss + delta*GL 
        total_loss.backward()
        optimizer.step()
        train_recon_loss_x1 += recon_loss_x1.item()
        train_recon_loss_x2 += recon_loss_x2.item()
        train_sam_loss += sam_loss.item() 
        train_gl_loss += GL.item()
        total_train_loss += (recon_loss_x2 + sam_loss + GL).item()

    writer.add_scalar("Loss/train", total_train_loss/len(train_loader), epoch)
    writer.add_scalar("Loss_GL/train", train_gl_loss/len(train_loader), epoch)
    writer.add_scalar("Loss_SAM/train", train_sam_loss/len(train_loader), epoch)
    # get eval loss every 2 epochs
    if epoch % 2 == 0:
        print(f"train epoch: {epoch} \
        recon_x1 loss: {train_recon_loss_x1/len(train_loader)}, \
        recon_x2 loss: {train_recon_loss_x2/len(train_loader)}, \
        sam_loss loss: {train_sam_loss/len(train_loader)}, \
        GL loss: {train_gl_loss/len(train_loader)}")
        model.eval()
        test_recon_loss_x2 = 0
        test_sam_loss = 0
        test_gl_loss = 0
        total_test_loss = 0
        for items in test_loader:
            c, y, z, x, lz = items
            x1, x2 = model(y, z)
            recon_loss_x1, recon_loss_x2, sam_loss, GL = model.calc_loss(x1, x2, lz, x)
            total_loss = recon_loss_x2 + sam_loss + GL 
            test_recon_loss_x2 += recon_loss_x2.item()
            test_sam_loss += sam_loss.item() 
            test_gl_loss += GL.item()
            total_test_loss = total_loss.item()

        print(f"test epoch: {epoch} \
        recon_x2 loss: {test_recon_loss_x2/len(test_loader)}, \
        sam_loss loss: {test_sam_loss/len(test_loader)}, \
        GL loss: {test_gl_loss/len(test_loader)}")
        total_test_loss /= len(test_loader)
        writer.add_scalar("Loss/test", total_test_loss/len(test_loader), epoch)
        writer.add_scalar("Loss_GL/test", test_gl_loss/len(test_loader), epoch)
        writer.add_scalar("Loss_SAM/test", test_sam_loss/len(test_loader), epoch)
    
        if best_model_test_loss > total_test_loss:
           best_model_test_loss =  total_test_loss
           print("saving ...")
           torch.save(model.state_dict(), model_path)
writer.close()
    

