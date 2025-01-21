import torch

from preliminary_course.ndarray.util import describe

if __name__ == '__main__':
    a = torch.arange(3).reshape(3, 1)
    b = torch.arange(2).reshape(1, 2)
    describe(a + b)