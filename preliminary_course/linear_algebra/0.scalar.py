import torch

if __name__ == '__main__':
    x = torch.tensor(3)
    y = torch.tensor(4)
    print(x + y)
    print(x * y)
    print(x / y)
    print(x ** y)  # power operation
    print(x % y)  # remainder