import torch

x = torch.arange(12)
print("x:", x)
print("x.shape:", x.shape)
print("x.numel:", x.numel())

x = x.reshape(3, 4)
print("x:", x)

