import torch

from preliminary_course.util import describe


def all_zeros():
    tensor_x = torch.zeros(2, 3, 4)
    describe(tensor_x)


if __name__ == '__main__':
    all_zeros()
