import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module import (Encoder, ResidualBlockNoBN, ResidualBlockNoBN3d,
                           CSFA, MFA, CBAM, InterpolateResize, make_layer)


class OurV1(nn.Module):

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 mid_channels=64,
                 mid_channels3d=8,
                 target_size=(20, 80, 80),
                 num_encoder_blocks=(4, 8, 12, 12),
                 num_decoder_blocks=(16, 12, 8, 4, 4),
                 res_scale=1.0):
        super().__init__()

        self.target_size = target_size
        self.mid_channels3d = mid_channels3d
        self.branches = len(num_decoder_blocks)

        self.first_conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.encoder = Encoder(
            mid_channels, ResidualBlockNoBN, num_encoder_blocks)

        self.bodies = nn.ModuleList()
        self.csfas = nn.ModuleList()
        self.up_samplings = nn.ModuleList()
        self.merge_blocks = nn.ModuleList()

        for i in range(self.branches):
            body = nn.ModuleList()
            for _ in range(i+1):
                body.append(make_layer(
                    ResidualBlockNoBN,
                    num_decoder_blocks[i],
                    mid_channels=mid_channels,
                    res_scale=res_scale))
            self.bodies.append(body)

        for i in range(self.branches - 1):
            self.csfas.append(CSFA(mid_channels, i+2, CBAM))
            self.merge_blocks.append(nn.Sequential(
                CBAM(gate_channels=mid_channels*2),
                nn.Conv2d(mid_channels*2, mid_channels, 3, 1, 1)))
            self.up_samplings.append(InterpolateResize(mid_channels))

        self.mfa = MFA(mid_channels, mid_channels3d *
                       target_size[0], self.branches, CBAM)

        self.res3d = make_layer(
            ResidualBlockNoBN3d,
            num_decoder_blocks[-1],
            mid_channels=mid_channels3d,
            res_scale=res_scale)

        self.last = nn.Conv3d(mid_channels3d, out_channels, 3, 1, 1)

    def forward(self, x):

        x0 = self.first_conv(x)
        tensors = self.encoder(x0)
        tensors.reverse()

        features = [tensors[0]]
        for i in range(self.branches):
            if i >= 1:
                if i < self.branches-1:
                    y = torch.cat([features[i], tensors[i]], dim=1)
                    features[i] = self.merge_blocks[i-1](y)
                features = self.csfas[i-1](features)
            for j in range(i + 1):
                features[j] = self.bodies[i][j](features[j])
            if i < self.branches-2:
                features.append(self.up_samplings[i](
                    features[-1], target_size=tensors[i+1].shape[2:]
                ))
            elif i == self.branches-2:
                features.append(self.up_samplings[i](
                    features[-1], target_size=self.target_size[-2:]
                ))
        x2d = self.mfa(features, target_size=self.target_size[-2:])

        b, c, h, w = x2d.shape
        d, ho, wo = self.target_size
        c_new = self.mid_channels3d
        x3d = x2d.view(b, c_new, d, h, w)
        # x3d = F.interpolate(x, size=self.target_size, mode='trilinear',
        #                     align_corners=False)

        x3d = self.res3d(x3d)
        out = self.last(x3d)

        return out


if __name__ == '__main__':
    x = torch.rand((2, 1, 40, 40)).cuda()
    model = CSFANet(1, 1).cuda()
    import time
    for i in range(10):
        y = model(x)
    t0 = time.time()
    for i in range(10):
        y = model(x)
    t1 = time.time()
    for i in range(20):
        y = model(x)
    t2 = time.time()
    print(t1-t0, t2-t1)
    torch.save(model.state_dict(), f'./tools/our_v1.pth')
    print(y.shape)
