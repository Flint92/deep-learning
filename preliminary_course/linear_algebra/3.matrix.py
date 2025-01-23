import torch

if __name__ == '__main__':
    A = torch.arange(12).reshape(3, 4)
    print(A)
    print(A.T) # 转置

    B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
    print(B == B.T) # 判断是否对称, 对称矩阵的转置等于自己