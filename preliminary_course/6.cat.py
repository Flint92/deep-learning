import torch

from preliminary_course.util import describe

if __name__ == '__main__':
    X = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    Y = torch.tensor([[[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]])
    describe(torch.cat((X, Y), dim=0))
    describe(torch.cat((X, Y), dim=1))
    describe(torch.cat((X, Y), dim=2))