import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import TensorDataset, DataLoader

def MSE(t1, t2):
    return F.mse_loss(t1, t2)

def fit(num_epochs, model, loss_fn, opt, train_dl):
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        if (epoch+1) % 10 == 0:
            print('Epoch[{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss))

inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58],
                   [102, 43, 37], [69, 96, 70], [73, 67, 43],
                   [91, 88, 64], [87, 134, 58], [102, 43, 37],
                   [69, 96, 70], [73, 67, 43], [91, 88, 64],
                   [87, 134, 58], [102, 43, 37], [69, 96,90]],
                   dtype='float32')

targets = np.array([[56, 70], [81, 101], [119, 133],
                   [22, 37], [103, 119], [56, 70],
                   [81, 101], [119, 133], [22, 37],
                   [103, 119], [56, 70], [81, 101],
                   [119, 133], [22, 37], [103, 119]],
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# Define dataset
print('Define dataset')
train_ds = TensorDataset(inputs, targets)
print(train_ds[0:3])
print('\n')

# Define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Define model
print('Define Model')
model = nn.Linear(3,2)
print(list(model.parameters()))
print('\n')

# Define Optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-5)

# Train Model
epochs = int(input('# of Epochs: '))
fit(epochs, model, MSE, opt, train_dl)

# Compare Prediction to Data
preds = model(inputs)
print(preds)
print(targets)