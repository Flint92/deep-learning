import torch

from preliminary_course.ndarray.util import describe


def reshape():
    tensor_x = torch.arange(12)
    tensor_x = tensor_x.reshape(3, 4)
    describe(tensor_x)


if __name__ == '__main__':
    reshape()
