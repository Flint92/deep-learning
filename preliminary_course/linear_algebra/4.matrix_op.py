import torch
from torch.utils.backcompat import keepdim_warning

if __name__ == '__main__':
    A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    B = A.clone()  # 通过分配新内存，将A的副本分配给B
    print(A)
    print(A + B)
    print(A * B) # 对应元素相乘, 不是矩阵乘法, 而是Hadamard积

    a = 2
    X = torch.arange(24).reshape(2, 3, 4)
    print(a + X)
    print((a * X).shape)

    A_sum_axis0 = A.sum(axis=0)
    print(A_sum_axis0)
    print(A_sum_axis0.shape)

    A_sum_axis1 = A.sum(axis=1)
    print(A_sum_axis1)
    print(A_sum_axis1.shape)

    A_sum = A.sum(axis=[0, 1]) # 所有维度求和等于求所有元素的和
    print(A_sum)
    print(A_sum.shape)

    print(A.mean()) # 求平均值
    print(A.numel()) # 元素个数

    print(A.sum(axis=1, keepdims=True)) # 保持维度
    print(A / A.sum(axis=1, keepdims=True)) # 广播机制

    print(A.cumsum(axis=0)) # 某个轴累积求和 running sum?

    x = torch.arange(4, dtype=torch.float32)
    print(torch.mv(A, x)) # 矩阵向量积

    B = torch.ones(4, 3)
    print(torch.mm(A, B)) # 矩阵乘法

    u = torch.tensor([3.0, -4.0])
    print(torch.norm(u)) # 范数
    print(torch.abs(u).sum()) # 绝对值求和
    print(torch.norm(torch.ones(4, 9))) # 矩阵范数