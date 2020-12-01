import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tt
import tarfile
import numpy as np
import matplotlib.pyplot as plt

# from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

from utils import *
from model import ResNet9

# Download from internet doesn't work
# dataset_url = "https://files.fast.ai/data/examples/cifar10.tgz"
# download_url(dataset_url, '.')

# with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
#     tar.extractall(path='./data')

if __name__ == '__main__':

    data_dir = './data/cifar10'
    classes = os.listdir(data_dir + '/train')

    # Data augmentation and normalization
    stats = ((0.4914, .4822, .4465), (0.2023, 0.1994, 0.2010))
    train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                            tt.RandomHorizontalFlip(),
                            tt.ToTensor(),
                            tt.Normalize(*stats,inplace=True)])

    valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

    # Datasets
    train_ds = ImageFolder(data_dir+'/train', train_tfms)
    valid_ds = ImageFolder(data_dir+'/test', valid_tfms)

    batch_size = 256

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size, num_workers=3, pin_memory=True)

    # show_batch(train_dl)

    device = get_default_device()

    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)

    # Train model
    model = to_device(ResNet9(3,10), device)

    history = [evaluate(model, valid_dl)]

    epochs = 10
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl,
                             grad_clip=grad_clip,
                             weight_decay=weight_decay,
                             opt_func=opt_func)

    torch.save(model.state_dict(), 'ResNet9.pth')

    plot_accuracies(history)
    plot_losses(history)
    plot_lrs(history)