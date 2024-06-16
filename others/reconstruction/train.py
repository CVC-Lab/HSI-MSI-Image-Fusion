from ..neural_nets.others import AutoEncoder
from datasets.cave_dataset import CAVEDataset, R
import numpy as np
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from .test_model import get_final_metric_scores
from .viz_utils import image_grid, plot_to_image
from accelerate import Accelerator
import torch.nn.functional as F
import torch
import os
import pdb
accelerator = Accelerator()
# torch.autograd.detect_anomaly()
## config
epochs = 300
batch_size = 2
in_channels = 31
dec_channels = 31
sf = 8
lr = 1e-4
model_name = "bg3"
load_best_model = False
model_path = f"./artifacts/{model_name}/gmodel.pth"
if not os.path.exists(f"./artifacts/{model_name}"):
    os.makedirs(f"./artifacts/{model_name}")

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.cuda.empty_cache()
device = accelerator.device
# device = torch.device("cuda:1")


writer = SummaryWriter(f"./artifacts/{model_name}")
train_dataset = CAVEDataset("./datasets/data/CAVE", None, mode="train")
test_dataset = CAVEDataset("./datasets/data/CAVE", None, mode="test")
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          shuffle=False, batch_size=batch_size)
model = AutoEncoder(in_channels=in_channels, dec_channels=dec_channels, R=R.to(device))#.cuda()
if load_best_model:
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
optimizer = Adam(model.parameters(), lr=lr)
print("starting training")


model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

alpha, beta, gamma, delta = 1, 1, 1.5, 1
best_model_test_loss = np.inf
n_random_samples = 2

for epoch in range(epochs):
    model.train()
    train_recon_loss_x = 0
    train_recon_loss_y = 0
    train_sam_loss = 0
    train_gl_loss = 0
    total_train_loss = 0
    for items in train_loader:
        optimizer.zero_grad()
        c, x_old, y, z, x, lz, idxs = items
        #x_old = x_old.cuda()
        # lz = lz.cuda()
        # y = y.cuda()
        z = z.to(device)
        lz = lz.to(device)
        y_hat, x_new = model(x_old, z, lz)
        
        for j, idx in enumerate(idxs):
            train_loader.dataset.x_states[int(idx.item())] = x_new[j, ...].detach().cpu()
        
        recon_loss_x, sam_loss, recon_loss_y, GL = model.calc_loss(x_new, y_hat, lz.detach(), x.to(device), y)
        total_loss = alpha*recon_loss_x + beta*recon_loss_y + gamma*sam_loss + delta*GL 
        # total_loss.backward()
        accelerator.backward(total_loss)
        optimizer.step()
        train_recon_loss_x += recon_loss_x.item()
        train_recon_loss_y += recon_loss_y.item()
        train_sam_loss += sam_loss.item() 
        train_gl_loss += GL.item()
        total_train_loss += (recon_loss_x + sam_loss + GL).item()

    writer.add_scalar("Loss/train", total_train_loss/len(train_loader), epoch)
    writer.add_scalar("Loss_GL/train", train_gl_loss/len(train_loader), epoch)
    writer.add_scalar("Loss_SAM/train", train_sam_loss/len(train_loader), epoch)
    # get eval loss every 2 epochs
    if epoch % 2 == 0:
        print(f"train epoch: {epoch} \
        recon_x loss: {train_recon_loss_x/len(train_loader)}, \
        recon_y loss: {train_recon_loss_y/len(train_loader)}, \
        sam_loss loss: {train_sam_loss/len(train_loader)}, \
        GLaplacian loss: {train_gl_loss/len(train_loader)}")
        # pdb.set_trace()
        # torch.cuda.set_device("cuda:0")
        model.eval()
        test_recon_loss_x = 0
        test_recon_loss_y = 0
        test_sam_loss = 0
        test_gl_loss = 0
        total_test_loss = 0
        with torch.no_grad():
            for items in test_loader:
                c, x_old, y, z, x_gt, lz, idx = items
                x_old = x_old.to(device)
                z = z.to(device)
                lz = lz.to(device)
                y_hat, x_new = model(x_old, z, lz)
                recon_loss_x, sam_loss, recon_loss_y, GL = model.calc_loss(x_new, y_hat, lz, x_gt.to(device), y.to(device))
                total_loss = recon_loss_x + sam_loss + GL 
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
        writer.add_scalar("Loss/test", total_test_loss/len(test_loader), epoch)
        writer.add_scalar("Loss_GL/test", test_gl_loss/len(test_loader), epoch)
        writer.add_scalar("Loss_SAM/test", test_sam_loss/len(test_loader), epoch)
    
        if best_model_test_loss > total_test_loss:
            best_model_test_loss =  total_test_loss
            print("saving ...")
            torch.save(model.state_dict(), model_path)
            opt = get_final_metric_scores(model, test_dataset, sf)
            writer.add_text("metrics", opt, epoch)
            img_tensor = []
            model.eval()
            for i in range(n_random_samples):
                items = test_dataset[i]
                c, x_old, y, z, x_gt, lz, idx = items
                z = z.to(device)
                lz =lz.to(device)

                y_hat, x_new = model(x_old[None, ...].to(device), z[None, ...], lz[None, ...])
                img_tensor.append(x_new.detach().cpu())

            for sid in range(n_random_samples):
                figure = image_grid(img_tensor[sid].squeeze(), sid)
                writer.add_image(f'{sid}_random_latent_space_samples_{epoch}',
                img_tensor=plot_to_image(figure)[..., :3],
                global_step=epoch,
                dataformats='HWC')

writer.close()
    

