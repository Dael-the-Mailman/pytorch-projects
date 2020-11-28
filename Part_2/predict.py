import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from model import MnistModel
from utils import *

def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

test_dataset = MNIST(root='data/', train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=256)

input_size = 28*28
num_classes = 10

model = MnistModel(input_size, num_classes)
model.load_state_dict(torch.load('./mnist-logistic.pth'))

img, label = test_dataset[np.random.randint(len(test_dataset))]
plt.imshow(img[0], cmap='gray')
plt.show()
print('Label:', label, ', Predicted:', predict_image(img, model))