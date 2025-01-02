import torch


def describe(tensor_x):
    print("x: {}".format(tensor_x))
    print("Shape: {}".format(tensor_x.shape))
    print("Size: {}".format(tensor_x.numel()))
    print("Type: {}\n".format(tensor_x.type()))


def reshape(tensor_x):
    tensor_x = tensor_x.reshape(3, 4)
    describe(tensor_x)


def all_zeros():
    tensor_x = torch.zeros(2, 3, 4)
    describe(tensor_x)


def all_ones():
    tensor_x = torch.ones(2, 3, 4)
    describe(tensor_x)


def assign_value():
    tensor_x = torch.tensor([
        [
            [2, 3, 4, 1],
            [1, 2, 3, 4],
            [4, 3, 2, 1]
        ]
    ])
    describe(tensor_x)


if __name__ == '__main__':
    x = torch.arange(12)
    describe(x)
    reshape(x)
    all_zeros()
    all_ones()
    assign_value()
