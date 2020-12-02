import os
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import tarfile

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from utils import *
from IPython.display import Image

# with tarfile.open('./data.tgz', 'r:gz') as tar:
#     tar.extractall(path='.')

if __name__ == '__main__':
    DATA_DIR = './data'

    image_size = 64
    batch_size = 128
    stats = (0.5,0.5,0.5), (0.5,0.5,0.5)

    train_ds = ImageFolder(DATA_DIR, transform=T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(*stats)
    ]))

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)

    device = get_default_device()
    latent_size = 128

    # show_batch(train_dl, stats) # Weeb stuff
    train_dl = DeviceDataLoader(train_dl, device)

    discriminator = to_device(nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),

        nn.Flatten(),
        nn.Sigmoid()
    ), device)

    generator = nn.Sequential(
        nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),

        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),

        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=0, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),

        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=0, bias=False),
        nn.Tanh()
    )

    xb = torch.randn(batch_size, latent_size, 1, 1)
    fake_images = generator(xb)
    show_images(fake_images, stats)

    generator = to_device(generator, device)

    fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
    save_samples(generator, stats, 0, fixed_latent)

    lr = 0.0002
    epochs = 25

    history = fit(discriminator, generator, train_dl, epochs, lr)
    Image('./generated/generated-images-0.00025.png')

    plt.plot(losses_d, '-')
    plt.plot(losses_g, '-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Discriminator', 'Generator'])
    plt.title('Losses');
    plt.show()
    plt.cla()

    plt.plot(real_scores, '-')
    plt.plot(fake_scores, '-')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend(['Real', 'Fake'])
    plt.title('Scores');
    plt.show()