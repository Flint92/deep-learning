import torch

from preliminary_course.util import describe


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
    assign_value()
