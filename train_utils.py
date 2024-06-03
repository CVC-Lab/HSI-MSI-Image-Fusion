import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_batch(data, device):
    # Original shape of hsi and rgb: (B, H, W, CH) and (B, H, W, 3)
    # Original shape of labels: (B, H, W, N)
    hsi_batch, rgb_batch, labels_batch = data

    # After conversion, hsi_batch and rgb_batch have shape (B, CH, H, W)
    # labels has shape (B, N, H, W)
    hsi_batch = np.moveaxis(hsi_batch, -1, 1)
    rgb_batch = np.moveaxis(rgb_batch, -1, 1)
    labels_batch = np.moveaxis(labels_batch, -1, 1)

    # Tensorize and put into device
    hsi_batch = torch.tensor(hsi_batch).to(device)
    rgb_batch = torch.tensor(rgb_batch).to(device)
    labels_batch = torch.tensor(labels_batch).to(device)


def main_training_loop(trainloader, net, optimizer, num_epochs,  
                       device=DEVICE, log_interval=100):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Prepare and tensorize batch.
            hsi_batch, rgb_batch, labels_batch = _prepare_batch(data, device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(hsi_batch, rgb_batch)
            loss = F.cross_entropy(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % log_interval == log_interval-1:    # print every log_interval mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / log_interval:.3f}')
                running_loss = 0.0

    print('Finished Training')


def test(testloader, net, device=DEVICE):
    """ Test via mIOU metrics. TBA"""
    total_correct = 0
    with torch.no_grad():
        for data in testloader:
            # Prepare and tensorize batch.
            hsi_batch, rgb_batch, labels_batch = _prepare_batch(data, device)
            outputs = net(hsi_batch, rgb_batch)
            _, predictions = torch.max(outputs, 1)
            total_correct += torch.sum(predictions == labels_batch).numpy()
        
    print('Total correct entry:', total_correct)