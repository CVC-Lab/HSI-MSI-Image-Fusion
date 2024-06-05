import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore
from losses import CombinedLoss

np.random.seed(0)
torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = CombinedLoss()

def main_training_loop(trainloader, net,
                       optimizer, scheduler, num_epochs, 
                       device=DEVICE, log_interval=5,
                       save_path='models/trained_model.pth'):
    ''' Main (standard) training loop'''
    lowest_loss = torch.inf
    ds_len = len(trainloader)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f'Epoch: {epoch+1}', end=' ')
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            hsi_batch, rgb_batch, labels_batch = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(hsi_batch.to(device), rgb_batch.to(device))
            loss = loss_fn(outputs, labels_batch.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        ep_loss = loss.item()/ds_len
        if epoch % log_interval == 0:
            print(f"loss:{ep_loss}")
        # save model if lowest loss
        if ep_loss < lowest_loss:
            lowest_loss = ep_loss
            print('saved')
            torch.save(net.state_dict(), save_path)
        scheduler.step(ep_loss)
    print('Done')


def test(testloader, net, save_path, num_classes, device=DEVICE):
    ''' Test via mIOU, GeneralizedDice metrics. TBA'''
    net.load_state_dict(torch.load(save_path))
    net.to(device)
    miou = MeanIoU(num_classes=num_classes, per_class=True)
    gdice = GeneralizedDiceScore(num_classes=num_classes, include_background=False)
    predictions = []
    truth_labels = []
    with torch.no_grad():
        for data in testloader:
            hsi_batch, rgb_batch, labels_batch = data
            outputs = net(hsi_batch.to(device), rgb_batch.to(device))
            predictions.append(torch.argmax(outputs.cpu(), axis=1))
            truth_labels.append(torch.argmax(labels_batch, axis=1))
    miou_score = miou(torch.cat(predictions, axis=0), torch.cat(truth_labels, axis=0)).numpy()
    gdice_score = gdice(torch.cat(predictions, axis=0), torch.cat(truth_labels, axis=0)).numpy()
    return miou_score, gdice_score
