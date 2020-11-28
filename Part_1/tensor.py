import torch
import numpy as np
import cv2 as cv
import pandas as pd

# Scalar
print("Scalar")
t1 = torch.tensor(4.)
print(t1)
print(t1.dtype)
print('\n')

# Vector
print("Vector")
t2 = torch.tensor([1., 2, 3, 4])
print(t2)
print('\n')

# Matrix
print("Matrix")
t3 = torch.tensor([[5.,6],[7,8], [9,10]])
print(t3)
print('\n')

# 3-D Array
print("3D Array")
t4 = torch.tensor([
    [[11, 12, 13],
     [13, 14, 15]],
    [[15, 16, 17],
     [17, 18, 19.]]])
print(t4)
print('\n')

# Shapes of Tensors
print("Shapes of Tensors")
print(t1.shape)
print(t2.shape)
print(t3.shape)
print(t4.shape)
print('\n')

# Tensor Operations & Gradients
x = torch.tensor(3.)
w = torch.tensor(4., requires_grad=True)
b = torch.tensor(5., requires_grad=True)

# Operations
print('Operations')
y = w * x + b
print(y)
print('\n')

# Compute Derivatives
print('Compute Derivatives')
print(y.backward())
print('\n')

# Gradients
print('Gradients')
print('dy/dx', x.grad)
print('dy/dw', w.grad)
print('dy/db', b.grad)
print('\n')

# Convert Numpy Array to Tensor
print('Convert Numpy Array to Tensor')
x = np.array([[1, 2], [3, 4.]])
y = torch.from_numpy(x)
print(y)
print(x.dtype, y.dtype)
print('\n')

# Convert Tensor to Array
print('Convert Tensor to Array')
z = y.numpy()
print(z)
print('\n')