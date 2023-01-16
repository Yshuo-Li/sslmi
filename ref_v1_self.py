import os
import time
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loss import TotalVariationLoss
from models import RefV1 as Model
from models import MagForward
from dataset import get_guide
from evaluate import psnr, ssim

start_epoch = 0
total_epoch = 3000

paddig = 3
in_channels = 20
size = (100, 62)
ano_size = (20, size[0] + paddig * 2, size[1] + paddig * 2)
zero_pad = nn.ZeroPad2d(3)

weight_forward = 1
weight_inversion = 0
weight_guide = 1
weight_tv = 1
weight_lambda = 1
tv_power = 2
lambda_type = 1

model_name = 'ref_v1_self_65'
work_dir = f'work_dirs/{model_name}'
load_from = 'work_dirs/ref_v1/weights/best.pth'


def train(
        load_from=None,
        in_path='data/field_data/field_data.mat',
        out_path='data/field_data/field_ref_v1_65.mat'):

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(torch.cuda.is_available())
    if not os.path.exists(f'./{work_dir}/weights/'):
        os.makedirs(f'./{work_dir}/weights/')
    inversion = Model(
        mid_channels=64,
        mid_channels3d=4,
        target_size=ano_size,
        num_encoder_blocks=(2, 4, 6, 8),
        num_decoder_blocks=(8, 6, 4, 2),
        num_3d_block=2,
        res_scale=1.0).to(device)
    forward = MagForward(in_channels, size, (65.66, 65.57, 100)).to(device)
    forward.load_state_dict(torch.load(
        'models/ckpt/mag_forward_100_62_20_real1.pth'))
    forward.eval()

    if load_from and os.path.exists(load_from):
        inversion.load_state_dict(torch.load(load_from))
        print('load inversion model')

    def lambda_loss_function(x):
        # print(tv_power, lambda_type)
        if lambda_type == 1:
            return torch.abs(x).mean()
        elif lambda_type == 2:
            return torch.pow(x, 2).mean()
    inversion_loss_function = nn.L1Loss()
    guide_loss_function = nn.L1Loss()
    forward_loss_function = nn.L1Loss()
    # lambda_loss_function = nn.MSELoss()
    tv_loss_function = TotalVariationLoss(1, power=tv_power)
    optimizer = optim.Adam(inversion.parameters(), lr=1e-4)
    scheduler = MultiStepLR(optimizer, [1000, 2000, 2500], gamma=0.5)

    ano = scio.loadmat(in_path)['ma'].astype(np.float32)
    ano_std = ano.std()
    # ano = ano / 400.  # 来自于数值范围的估计
    ano /= ano_std
    ano = torch.from_numpy(ano[np.newaxis, np.newaxis, ...]).to(device)

    pad_ano = zero_pad(ano)
    g = torch.tensor([[0, 0.3, 0.8, 1, 0.9,
                       0.7, 0.6, 0.5, 0.4, 0.3,
                       0.2, 0.1, 0, 0, 0,
                       0, 0, 0, 0, 0
                       ]]).to(device)
    g = g/g.sum(dim=-1, keepdim=True)*g.shape[-1]/2

    min_loss = 100
    log_path = f'./{work_dir}/weights/log.txt'
    writer = SummaryWriter(f"./{work_dir}/weights/log")
    for _ in range(start_epoch):
        optimizer.step()
        scheduler.step()
    t0 = time.time()
    for i in range(start_epoch, total_epoch):
        inversion.train()
        loss_forward_sum = 0
        loss_inversion_sum = 0
        loss_guide_sum = 0
        loss_tv_sum = 0

        pad_body_ = inversion(pad_ano, g)
        body_ = pad_body_[..., paddig:-paddig, paddig:-paddig]
        g_ = get_guide(body_)
        # ano_ = forward(body_) / 400  # 来自于数值范围的估计
        ano_ = forward(body_) / ano_std  # 来自于数值范围的估计
        optimizer.zero_grad()
        # print(ano.shape, ano.min(), ano.max())
        # print(ano_gt.shape, ano_gt.min(), ano_gt.max())
        # print(body_.shape, body_.min(), body_.max())
        # print(ano_.shape, ano_.min(), ano_.max())
        # print(g.shape, g.min(), g.max())
        # input()

        loss_forward = forward_loss_function(ano_, ano)
        loss_guide = guide_loss_function(g_, g)
        loss_lambda = lambda_loss_function(pad_body_)
        loss_tv = tv_loss_function(pad_body_)
        if i < 400:
            loss = loss_guide*50 + loss_tv * weight_tv + loss_forward
        elif i < 2500:
            loss = (loss_forward * weight_forward +
                    loss_lambda * weight_lambda +
                    loss_guide * weight_guide +
                    loss_tv * weight_tv)
        else:
            loss = (loss_forward * weight_forward +
                    loss_lambda * weight_lambda * 5 +
                    loss_guide * weight_guide +
                    loss_tv * weight_tv * 5)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if loss_forward < min_loss*0.99 and i > 1500:
            min_loss = loss_forward
            torch.save(inversion.state_dict(),
                       f'./{work_dir}/best.pth')

        learn_rate = optimizer.state_dict()['param_groups'][0]['lr']
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f'inference {i:04d}\tloss: {loss:.6f}'
                    f'\tforward {loss_forward:.6f}'
                    f'\tguide {loss_guide:.6f}'
                    f'\tlambda {loss_lambda:.6f}'
                    f'\ttv {loss_tv:.6f}'
                    f'\tlr {learn_rate:.6f}\n')
        if i % 50 == 0:
            print(time.time()-t0)
            t0 = time.time()
            print(f'inference {i:04d}\tloss: {loss:.6f}'
                  f'\tforward {loss_forward:.6f}'
                  f'\tguide {loss_guide:.6f}'
                  f'\tlambda {loss_lambda:.6f}'
                  f'\ttv {loss_tv:.6f}'
                  f'\tlr {learn_rate:.6f}')
            if i % 200 == 0:
                print('body_', body_.shape, body_.min(), body_.max())
                print('ano', ano.shape, ano.min(), ano.max(), ano.mean())
                print('ano_', ano_.shape, ano_.min(), ano_.max())
                print('g_', g_.shape, g_.min(), g_.max())

            body_ = body_[0, 0].permute(1, 2, 0).detach().cpu().numpy()
            ano_ = ano_[0, 0].detach().cpu().numpy()
            g_ = g_[0].detach().cpu().numpy()

            scio.savemat(out_path, dict(mag=body_, g=g_, ma=ano_))
            plt.imsave(out_path + '.png', ano_, cmap='jet')


def inference(
        load_from=load_from,
        in_path='data/field_data/field_data.mat',
        out_path='data/field_data/field_ref_v1.mat'):

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(torch.cuda.is_available())
    if not os.path.exists(f'./{work_dir}/weights/'):
        os.makedirs(f'./{work_dir}/weights/')
    inversion = Model(
        mid_channels=64,
        mid_channels3d=4,
        target_size=ano_size,
        num_encoder_blocks=(2, 4, 6, 8),
        num_decoder_blocks=(8, 6, 4, 2),
        num_3d_block=2,
        res_scale=1.0).to(device)
    forward = MagForward(in_channels, size).to(device)
    forward.load_state_dict(torch.load(
        'models/ckpt/mag_forward_100_62_20.pth'))
    inversion.eval()
    forward.eval()

    if load_from and os.path.exists(load_from):
        inversion.load_state_dict(torch.load(load_from))
        print('load inversion model')

    ano = scio.loadmat(in_path)['ma'].astype(np.float32)
    ano_std = ano.std()
    # ano = ano / 400.  # 来自于数值范围的估计
    ano /= ano_std
    ano = torch.from_numpy(ano[np.newaxis, np.newaxis, ...]).to(device)

    pad_ano = zero_pad(ano)
    g = torch.tensor([[0, 0.3, 0.8, 1, 0.9,
                       0.7, 0.6, 0.5, 0.4, 0.3,
                       0.2, 0.1, 0, 0, 0,
                       0, 0, 0, 0, 0
                       ]]).to(device)
    g = g/g.sum(dim=-1, keepdim=True)*g.shape[-1]/2

    pad_body_ = inversion(pad_ano, g)
    body_ = pad_body_[..., paddig:-paddig, paddig:-paddig]
    g_ = get_guide(body_)

    body_ = body_[0, 0].permute(1, 2, 0).detach().cpu().numpy()
    ano_ = ano_[0, 0].detach().cpu().numpy()
    g_ = g_[0].detach().cpu().numpy()

    scio.savemat(out_path, dict(mag=body_, g=g_, ma=ano_))
    plt.imsave(out_path + '.png', ano_, cmap='jet')


if __name__ == '__main__':
    train()
