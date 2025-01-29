import torch
from IPython import display
from d2l import torch as d2l

num_inputs = 784  # because of 28x28 images
num_outputs = 10  # because of dataset has 10 different labels

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = torch.sum(X_exp, dim=1, keepdim=True)
    return X_exp / partition # 广播机制


def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def eval_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式
    metric = Accumulator(2) # 正确预测数，预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    X = torch.normal(0, 1, (2, 5))
    X_prob = softmax(X)
    print(X_prob)
    print(X_prob.sum(dim=1))

    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    print(cross_entropy(y_hat, y))
    print(accuracy(y_hat, y) / len(y))
    print(eval_accuracy(net, test_iter))

    lr = 0.1
    def updater(batch_size):
        return d2l.sgd([W, b], lr, batch_size)


    num_epochs = 10
    d2l.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

    d2l.predict_ch3(net, test_iter)

    d2l.plt.show()