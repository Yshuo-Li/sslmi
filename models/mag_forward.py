import time
import os
from tqdm import tqdm

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR


def calculate(i, j, k, m, n):

    i, j, k, m, n = i + 1, j + 1, k + 1, m + 1, n + 1

    part1 = np.arctan(
        (m - (i + 0.5)) * (n - (j + 0.5)) /
        (((m - (i + 0.5))**2 + (n - (j + 0.5))**2 +
          (k + 0.5)**2)**(0.5)) / (k + 0.5))
    part2 = np.arctan(
        (m - (i - 0.5)) * (n - (j + 0.5)) /
        (((m - (i - 0.5))**2 + (n - (j + 0.5))**2 +
          (k + 0.5)**2)**(0.5)) / (k + 0.5))
    part3 = np.arctan(
        (m - (i + 0.5)) * (n - (j - 0.5)) /
        (((m - (i + 0.5))**2 + (n - (j - 0.5))**2 +
          (k + 0.5)**2)**(0.5)) / (k + 0.5))
    part4 = np.arctan(
        (m - (i - 0.5)) * (n - (j - 0.5)) /
        (((m - (i - 0.5))**2 + (n - (j - 0.5))**2 +
          (k + 0.5)**2)**(0.5)) / (k + 0.5))
    part5 = np.arctan(
        (m - (i + 0.5)) * (n - (j + 0.5)) /
        (((m - (i + 0.5))**2 + (n - (j + 0.5))**2 +
          (k - 0.5)**2)**(0.5)) / (k - 0.5))
    part6 = np.arctan(
        (m - (i - 0.5)) * (n - (j + 0.5)) /
        (((m - (i - 0.5))**2 + (n - (j + 0.5))**2 +
          (k - 0.5)**2)**(0.5)) / (k - 0.5))
    part7 = np.arctan(
        (m - (i + 0.5)) * (n - (j - 0.5)) /
        (((m - (i + 0.5))**2 + (n - (j - 0.5))**2 +
          (k - 0.5)**2)**(0.5)) / (k - 0.5))
    part8 = np.arctan(
        (m - (i - 0.5)) * (n - (j - 0.5)) /
        (((m - (i - 0.5))**2 + (n - (j - 0.5))**2 +
          (k - 0.5)**2)**(0.5)) / (k - 0.5))

    return part1 - part2 - part3 + part4 - part5 + part6 + part7 - part8


cache = dict()


def fast_calculate(i, j, k, m, n, s_i=100, s_j=100, s_k=100):
    i, j, k = m - i, n - j, k + 1
    s_i, s_j = s_i / s_k, s_j / s_k
    params = f'{i}_{j}_{k}_{s_i:03f}_{s_j:03f}'
    if not params in cache:
        cache[params] = cache_calculate(i, j, k, s_i, s_j)
    return cache[params]


def cache_calculate(i, j, k, s_i, s_j):
    """
    i: m - i
    j: n - j
    """

    part1 = np.arctan(
        (i - 0.5) * (j - 0.5) * s_i * s_j /
        (((i - 0.5)**2 * s_i**2 + (j - 0.5)**2 * s_j**2 +
          (k + 0.5)**2)**(0.5)) / (k + 0.5))
    part2 = np.arctan(
        (i + 0.5) * (j - 0.5) * s_i * s_j /
        (((i + 0.5)**2 * s_i**2 + (j - 0.5)**2 * s_j**2 +
          (k + 0.5)**2)**(0.5)) / (k + 0.5))
    part3 = np.arctan(
        (i - 0.5) * (j + 0.5) * s_i * s_j /
        (((i - 0.5)**2 * s_i**2 + (j + 0.5)**2 * s_j**2 +
          (k + 0.5)**2)**(0.5)) / (k + 0.5))
    part4 = np.arctan(
        (i + 0.5) * (j + 0.5) * s_i * s_j /
        (((i + 0.5)**2 * s_i**2 + (j + 0.5)**2 * s_j**2 +
          (k + 0.5)**2)**(0.5)) / (k + 0.5))
    part5 = np.arctan(
        (i - 0.5) * (j - 0.5) * s_i * s_j /
        (((i - 0.5)**2 * s_i**2 + (j - 0.5)**2 * s_j**2 +
          (k - 0.5)**2)**(0.5)) / (k - 0.5))
    part6 = np.arctan(
        (i + 0.5) * (j - 0.5) * s_i * s_j /
        (((i + 0.5)**2 * s_i**2 + (j - 0.5)**2 * s_j**2 +
          (k - 0.5)**2)**(0.5)) / (k - 0.5))
    part7 = np.arctan(
        (i - 0.5) * (j + 0.5) * s_i * s_j /
        (((i - 0.5)**2 * s_i**2 + (j + 0.5)**2 * s_j**2 +
          (k - 0.5)**2)**(0.5)) / (k - 0.5))
    part8 = np.arctan(
        (i + 0.5) * (j + 0.5) * s_i * s_j /
        (((i + 0.5)**2 * s_i**2 + (j + 0.5)**2 * s_j**2 +
          (k - 0.5)**2)**(0.5)) / (k - 0.5))

    return part1 - part2 - part3 + part4 - part5 + part6 + part7 - part8


def mag_forward_old(data, expect_size=(40, 40)):
    h, w, d = data.shape[-3:]
    ho, wo = expect_size
    magnetism = torch.zeros(data.shape[:-3] + expect_size).cuda()
    t0 = time.time()
    for m in tqdm(range(ho)):
        for n in range(wo):
            mag = 0
            for i in range(h):
                for j in range(w):
                    for k in range(d):
                        # print(i, j, k, m, n)
                        mag += fast_calculate(i, j, k, m, n) * data[...,
                                                                    i, j, k]
            magnetism[..., m, n] = 0 - mag * 4000
    # print(gravity[0, 0], gravity.dtype)
    return magnetism


class MagForwardConv(nn.Conv2d):
    def __init__(self, in_channels, kernel_size, set_params=True):
        kernel_size = (kernel_size, kernel_size) if isinstance(
            kernel_size, int) else kernel_size
        padding = [k//2 for k in kernel_size]
        super().__init__(in_channels, 1,
                         kernel_size=kernel_size,
                         stride=1,
                         padding=padding,
                         bias=False)
        if set_params:
            weight = self.weight.data
            c1, c2 = kernel_size[0]//2, kernel_size[1]//2
            for i in range(kernel_size[0]):
                for j in range(kernel_size[1]):
                    for k in range(in_channels):
                        weight[0, k, i, j] = calculate(i, j, k, c1, c2)
            self.weight.data = weight * 40000 / 10


class MagForward(nn.Module):
    def __init__(self, in_channels, size, sample=(100, 100, 100), set_params=False):
        super().__init__()

        h, w = size
        d = in_channels
        s_i, s_j, s_k = sample
        print(s_i, s_j, s_k)
        self.in_features = h * w * d
        self.out_features = h * w
        self.out_size = size
        self.layer = nn.Linear(self.in_features, self.out_features, bias=False)
        self.layer.requires_grad = False

        if set_params:
            weight = self.layer.weight.data.reshape(h, w, d, h, w)
            for m in tqdm(range(h)):
                for n in range(w):
                    for k in range(d):
                        for i in range(h):
                            for j in range(w):
                                weight[m, n, k, i, j] = - fast_calculate(
                                    i, j, k, m, n,
                                    s_i, s_j, s_k)
                                # weight[m, n, k, i, j] = -1
            self.layer.weight.data = weight.reshape(
                self.out_features, self.in_features) * 40000 / 10

    def forward(self, x):
        b, c, d, h, w = x.shape
        # print(x.shape)
        x = x.contiguous().view(b, d*h*w)
        # print(x.shape, self.layer.weight.shape)
        y = self.layer(x)
        y = y.view(b, 1, self.out_size[-2], self.out_size[-1])
        return y


def run():
    import matplotlib.pyplot as plt

    mag = mag_c = scio.loadmat('data/magnetic_data/magnetic_1.mat')['mag']
    anomaly = scio.loadmat('data/magnetic_data/magnetic_anomaly_1.mat')['ma']
    plt.imsave('data/field_data/mag_1_anomaly.png', anomaly)
    print(mag.shape, mag.max(), mag.min())
    plt.imsave('data/field_data/mag_1_mag80.png', mag[:, :, 10])
    mag = mag[1::2, 1::2].astype(np.float32).transpose((2, 0, 1))
    plt.imsave('data/field_data/mag_1_mag40.png', mag[10, :, :])
    print(mag.shape, mag.max(), mag.min())
    print(anomaly.shape, anomaly.max(), anomaly.min())
    mag = torch.from_numpy(mag[np.newaxis, ::2, ...])
    print(mag.shape, mag.max(), mag.min())
    linear = MagForward(10, (40, 40), set_params=True)
    torch.save(linear.state_dict(), 'models/ckpt/mag_forward_40_20.pth')
    # linear.load_state_dict(torch.load('models/ckpt/mag_forward_40_20.pth'))
    ano = linear(mag)
    ano = ano[0, 0].cpu().detach().numpy()
    print(ano.shape, ano.max(), ano.min())
    # ano = ano / 1000
    # print(ano.shape, ano.max(), ano.min())
    scio.savemat('data/field_data/mag_1.mat',
                 dict(mag=mag_c, ma=anomaly, pred_ma=ano))


def check():
    import matplotlib.pyplot as plt
    data = scio.loadmat('data/field_data/mag_1.mat')
    plt.imsave('data/field_data/mag_1_pred.png', data['pred_ma'])
    print(data.keys())


def test():
    ans = calculate(1, 1, 1, 1, 1)
    print(ans)


def create():
    linear = MagForward(20, (100, 62), (65.66, 65.57, 100), set_params=True)
    torch.save(linear.state_dict(),
               'models/ckpt/mag_forward_100_62_20_real1.pth')


if __name__ == '__main__':
    # run()
    # check()
    # test()
    create()

    # # data = torch.rand(1, 1, 80, 80, 2).cuda()
    # # # torch.save(data, 'data.pth')
    # # # data = torch.load('data.pth')
    # # magnetism = mag_forward_old(data, expect_size=(80, 80))
    # # torch.save(dict(magnetism=magnetism, data=data), 'magnetism.pth')
    # d = torch.load('magnetism.pth')
    # magnetism = d['magnetism']
    # data = d['data']
    # # print(magnetism)

    # data = torch.rand(1, 1, 18, 18, 2).cuda()
    # magnetism = mag_forward_old(data, expect_size=(18, 18))

    # # conv = MagForwardConv(2, 83).cuda()
    # # trans_data = data[0].permute(0, 3, 1, 2)
    # # mag_conv = conv(trans_data)

    # linear = MagForward(20, (40, 40), set_params=True)
    # torch.save(linear.state_dict(), 'models/ckpt/mag_forward_40_20.pth')
    # trans_data = data[0].permute(0, 3, 1, 2)
    # mag_linear = linear(trans_data)

    # diff = abs(mag_linear-magnetism).sum()
    # print(diff)
    # print(diff/magnetism.sum())
