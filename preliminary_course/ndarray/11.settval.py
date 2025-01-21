import torch

from preliminary_course.ndarray.util import describe

if __name__ == '__main__':
    X = torch.arange(12, dtype=torch.float32).reshape( 3, 4)
    X[1,2]=9
    describe(X)
    X[1:2, :]=12
    describe(X)