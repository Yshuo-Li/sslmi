import torch
import torch.nn as nn


class TotalVariationLoss(nn.Module):

    def __init__(self, weight=1., power=1):
        super().__init__()

        self.weight = weight
        self.power = power

    def forward(self, x):
        if len(x.shape) == 2:
            tv = torch.pow(torch.abs(x[:, 1:] - x[:, :-1]), self.power).mean()
        elif len(x.shape) <= 4:
            tv_h = torch.pow(torch.abs(x[:, :, 1:, :] -
                             x[:, :, :-1, :]), self.power).mean()
            tv_w = torch.pow(torch.abs(x[:, :, :, 1:] -
                             x[:, :, :, :-1]), self.power).mean()
            tv = tv_h + tv_w
        elif len(x.shape) == 5:
            tv_d = torch.pow(torch.abs(x[:, :, 1:, :, :] -
                             x[:, :, :-1, :, :]), self.power).mean()
            tv_h = torch.pow(torch.abs(x[:, :, :, 1:, :] -
                             x[:, :, :, :-1, :]), self.power).mean()
            tv_w = torch.pow(torch.abs(x[:, :, :, :, 1:] -
                             x[:, :, :, :, :-1]), self.power).mean()
            tv = tv_d + tv_h + tv_w
        return self.weight * tv
