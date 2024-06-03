import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.segmentation import MeanIoU


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main_training_loop(trainloader, net,
                       optimizer, num_epochs, 
                       device=DEVICE, log_interval=100,
                       save_path='models/trained_model.pth'):
    ''' Main (standard) training loop'''
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f'\nEpoch: {epoch+1}')
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            hsi_batch, rgb_batch, labels_batch = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(hsi_batch.to(device), rgb_batch.to(device))
            loss = F.cross_entropy(outputs, labels_batch.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % log_interval == log_interval-1:    # print every log_interval mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / log_interval:.3f}')
                running_loss = 0.0

    print('Finished Training. Saving model...')
    torch.save(net.state_dict(), save_path)
    print('Done')


def test(testloader, net, num_classes, device=DEVICE):
    ''' Test via mIOU metrics. TBA'''
    miou = MeanIoU(num_classes=num_classes, per_class=True)
    predictions = []
    truth_labels = []
    with torch.no_grad():
        for data in testloader:
            hsi_batch, rgb_batch, labels_batch = data
            outputs = net(hsi_batch.to(device), rgb_batch.to(device))
            predictions.append(torch.argmax(outputs.cpu(), axis=1))
            truth_labels.append(torch.argmax(labels_batch, axis=1))

    return miou(torch.cat(predictions, axis=0), torch.cat(truth_labels, axis=0)).numpy()
