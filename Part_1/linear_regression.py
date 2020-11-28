import numpy as np
import torch

inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

# Data Tensors
print('Data Tensors')
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print(inputs)
print(targets)
print('\n')

# Weights and Biases
print('Weights and Biases')
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
print(w)
print(b)
print('\n')

# Linear Regression Model
def model(x):
    return x @ w.t() + b

# Initial Predictions
print('Initial Predictions')
preds = model(inputs)
print('Prediction: ', preds)
print('Target: ', targets)
print('\n')

# Loss Function
print('Compute Loss')
def MSE(t1, t2):
    diff = t1 - t2
    return torch.sum(diff*diff)/diff.numel()
loss = MSE(preds, targets)
print(loss)
print('\n')

# Compute Gradients
print('Compute Gradients')
loss.backward()
print(w)
print(w.grad)
print('\n')

# Reset Gradients
print('Reset Gradients')
w.grad.zero_()
b.grad.zero_()
print(w,b)
print(w.grad)
print(b.grad)
print('\n')

# Train for N-Epochs
print('Train for N-Epochs')
epochs = int(input('# of Epochs: '))
lr = 1e-5
for i in range(epochs):
    preds = model(inputs)
    loss = MSE(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad*lr
        b -= b.grad*lr
        w.grad.zero_()
        b.grad.zero_()
print('Training Finished!\n')

# Calculate New Loss
print('Calculate New Loss')
preds = model(inputs)
loss = MSE(preds, targets)
print(loss)
print('\n')

# Compare Predictions to Data
print('Compare Predictions to Data')
print('Prediction: ', preds)
print('Target: ', targets)