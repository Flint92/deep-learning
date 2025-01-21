import torch

from preliminary_course.ndarray.util import describe


def arithmetic():
    tensor_x = torch.tensor([1.0, 2, 4, 8])
    tensor_y = torch.tensor([2, 2, 2, 2])
    describe(tensor_x + tensor_y)
    describe(tensor_x - tensor_y)
    describe(tensor_x * tensor_y)
    describe(tensor_x / tensor_y)
    describe(tensor_x ** tensor_y) # power operation


if __name__ == '__main__':
    arithmetic()
