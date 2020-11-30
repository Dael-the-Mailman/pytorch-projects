import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from model import Cifar10CNNModel
from utils import predict_image, to_device, get_default_device

device = get_default_device()

model = Cifar10CNNModel()
model.load_state_dict(torch.load('./Cifar10CNNModel.pth'))
to_device(model, device)

model.eval()

dataset = ImageFolder('./data/cifar10/test', transform=ToTensor())
img, label = dataset[np.random.randint(len(dataset))]
plt.imshow(img.permute(1,2,0))
print('Label:', dataset.classes[label], ', Predicted:', 
       predict_image(img, model, dataset.classes, device))
plt.show()