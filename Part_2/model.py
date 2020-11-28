import torch
import torch.nn as nn

from utils import *

# Logistic Regression Model
class MnistModel(nn.Module):
    def __init__(self, inp_size, num_class):
        super().__init__()
        self.linear = nn.Linear(inp_size, inp_size)

    def forward(self, xb):
        xb = xb.reshape(-1,784)
        out = self.linear(xb)
        return out 

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = loss_fn(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = loss_fn(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print('Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}'.format(epoch, result['val_loss'], result['val_acc']))

model = MnistModel(784, 10)