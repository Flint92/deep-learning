import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from d2l import torch as d2l

d2l.use_svg_display()

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    d2l.plt.show()


def get_dataloader_workers():
    """使用4个进程读取数据"""
    return 4


if __name__ == '__main__':
    # 将图像数据从PIL类型转化为float32类型
    # 并除以255使得所有像素的数值均在0到1之间
    trans = transforms.ToTensor()

    mnist_train = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=trans, download=True
    )

    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trans, download=True
    )

    print(len(mnist_train), len(mnist_test))
    print(mnist_train[0][0].shape)

    X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    show_images(X.reshape(18, 28, 28), 2, 10, titles=get_fashion_mnist_labels(y))

    # batch_size = 256
    # train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers())
    #
    # timer = d2l.Timer()
    # for X, y in train_iter:
    #     continue
    # print(f'{timer.stop():.2f} sec')

