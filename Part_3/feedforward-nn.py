import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from model import MnistModel
from utils import *

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

if __name__ == '__main__':
    # Create Datasets
    dataset = MNIST(root='./data/', download=True, transform=ToTensor())

    val_size = 10000
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Training Variables
    batch_size = 128

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

    # Visualize Data
    for images, _ in train_loader:
        print('images.shape:', images.shape)
        plt.figure(figsize=(16,9))
        plt.axis('off')
        plt.imshow(make_grid(images, nrow=16).permute((1,2,0)))
        # plt.savefig('the-numbers-mason.png', pad_inches=0.0)
        plt.show()
        plt.clf()
        break

    input_size = 784
    hidden_size = 32
    num_classes = 10
    model = MnistModel(input_size, hidden_size, num_classes)
    
    # Load Model and Data on GPU
    device = get_default_device()
    to_device(model, device)
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)

    history = [evaluate(model, val_loader)]
    history += fit(5, 0.5, model, train_loader, val_loader)
    history += fit(5, 0.1, model, train_loader, val_loader)

    # Plot result
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of Epochs')
    plt.savefig('Loss.png')
    plt.show()

    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of Epochs')
    plt.savefig('Accuracy.png')
    plt.show()
