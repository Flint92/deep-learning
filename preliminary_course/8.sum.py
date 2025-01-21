import torch

from preliminary_course.util import describe

if __name__ == '__main__':
    X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    describe(X.sum()) # scalar