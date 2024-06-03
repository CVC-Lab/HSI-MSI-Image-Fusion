import torch
import torch.nn.functional as F
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main_training_loop(trainloader, net,
                       optimizer, num_epochs, 
                       device=DEVICE, log_interval=100):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Prepare and tensorize batch.
            hsi_batch, rgb_batch, labels_batch = data
            hsi_batch, rgb_batch, labels_batch = (
                torch.tensor(hsi_batch.to(device), dtype=torch.float),
                torch.tensor(rgb_batch.to(device), dtype=torch.float),
                torch.tensor(labels_batch.to(device), dtype=torch.float)
            )

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            print(hsi_batch.shape, rgb_batch.shape)
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
            hsi_batch, rgb_batch, labels_batch = data
            outputs = net(hsi_batch, rgb_batch)
            _, predictions = torch.max(outputs, 1)
            total_correct += torch.sum(predictions == labels_batch).numpy()
        
    print('Total correct entry:', total_correct)