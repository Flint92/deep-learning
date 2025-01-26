import torch

if __name__ == '__main__':
    x = torch.arange(4.0)
    x.requires_grad_(True) # Same as `x = torch.arange(4.0, requires_grad=True)`
    print(x.grad) # None

    y = 2 * torch.dot(x, x)
    print(y)

    y.backward()
    print(x.grad == 4 * x) # True

    x.grad.zero_() # Clear the previous value of x.grad
    y = x.sum()
    y.backward()
    print(x.grad) # 1 1 1 1

    x.grad.zero_() # Clear the value of x.grad
    y = x * x
    y.sum().backward() # Equivalent to y.backward(torch.ones(len(x))) or y.backward(torch.ones(y.shape))
    print(x.grad) # [0, 2, 4, 6]