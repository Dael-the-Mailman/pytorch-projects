import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

from model import ResNet9
from utils import get_default_device, to_device, predict_image

device = get_default_device()

model = to_device(ResNet9(3,10),device)
model.load_state_dict(torch.load('./ResNet9.pth'))
to_device(model, device)

model.eval()

dataset = ImageFolder('./data/cifar10/test', transform=ToTensor())
img, label = dataset[np.random.randint(len(dataset))]
plt.imshow(img.permute(1,2,0))
print('Label:', dataset.classes[label], ', Predicted:', 
       predict_image(img, model, dataset.classes, device))
plt.show()