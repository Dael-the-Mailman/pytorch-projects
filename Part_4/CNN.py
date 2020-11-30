import os 
import torch
import torchvision
import tarfile
import matplotlib.pyplot as plt

from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from utils import *
from model import Cifar10CNNModel

# Download Data (Doesn't work for some reason)
# dataset_url = 'https://files.fast.ai/data/examples/cifar10.tgz'
# download_url(dataset_url, '.')

# Extract from tarfile
# with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
#     tar.extractall(path="./data")

data_dir = './data/cifar10'

dataset = ImageFolder(data_dir+'/train', transform=ToTensor())

torch.manual_seed(42)

# Split Data
val_size = 5000
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

batch_size = 128
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

model = Cifar10CNNModel()

device = get_default_device()
if __name__ == '__main__':
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(model, device)

    num_epochs = 10
    opt_func = torch.optim.Adam
    lr = 0.001

    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

    plot_accuracies(history)
    plot_losses(history)

    torch.save(model.state_dict(), 'Cifar10CNNModel.pth')