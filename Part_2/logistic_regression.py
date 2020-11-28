import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from model import MnistModel
from utils import *

# Create Dataset
# dataset = MNIST(root='data/', download=True)
dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())
test_dataset = MNIST(root='data/', train=False, transform=transforms.ToTensor())

# Split Dataset into training and validation sets
train_ds, val_ds = random_split(dataset, [50_000, 10_000])

# Training Variables
batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

input_size = 28*28
num_classes = 10

# Create model
model = MnistModel(input_size, num_classes)

# Train model
history = fit(30, 0.001, model, train_loader, val_loader)

# Plot model
accuracies = [result['val_acc'] for result in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of Epochs')
plt.show()

# Evaluate Model on Test Data
print('Evaluate Model on Test Data')
test_loader = DataLoader(test_dataset, batch_size=256)
result = evaluate(model, test_loader)
print(result)
print('\n')

torch.save(model.state_dict(), 'mnist-logistic.pth')