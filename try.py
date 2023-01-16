import torch
from scipy.io import loadmat


if __name__ == '__main__':
    # x = torch.rand((2, 1, 40, 40)).cuda()
    # l = torch.rand((2, 20)).cuda()
    # model = OurV1(1, 1).cuda()
    # y = model(x, l)
    # print(y.shape)
    data = loadmat('data/field_data/mag_1.mat')
    mag = data['mag']
    ma = data['ma']
    print(mag.shape, mag.min(), mag.max())
    print(ma.shape, ma.min(), ma.max())
