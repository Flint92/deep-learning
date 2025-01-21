import torch

from preliminary_course.util import describe

if __name__ == '__main__':
    X = torch.arange(12, dtype=torch.float32).reshape( 3, 4)
    describe(X[-1]) #  最后一行
    describe(X[1:3]) # 第二行和第三行，注意这里的切片是左闭右开的