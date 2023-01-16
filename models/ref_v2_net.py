import torch
import torch.nn as nn
import torch.nn.functional as F

from . import RefV1
from .module import (Encoder, ResidualBlockNoBN, ResidualBlockNoBN3d,
                     CSFA, MFA, CBAM, InterpolateResize, make_layer,
                     Expander)


class RefV2(RefV1):

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 mid_channels=64,
                 mid_channels3d=8,
                 target_size=(20, 80, 80),
                 num_encoder_blocks=(4, 8, 12, 12),
                 num_decoder_blocks=(16, 12, 8, 4),
                 num_3d_block=4,
                 res_scale=1.0):

        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         mid_channels=mid_channels,
                         mid_channels3d=mid_channels3d,
                         target_size=target_size,
                         num_encoder_blocks=num_encoder_blocks,
                         num_decoder_blocks=num_decoder_blocks,
                         num_3d_block=num_3d_block,
                         res_scale=res_scale)

        self.conv3d = nn.Conv3d(mid_channels3d, mid_channels3d, 3, 1, 1)

    def forward(self, x, g):

        x0 = self.first_conv(x)
        tensors = self.encoder(x0)
        tensors.reverse()
        guides = self.expander(g, tensors)

        feature = torch.cat([tensors[0], guides[0]], dim=1)
        features = [self.first_merge(feature)]
        for i in range(self.branches):
            if i >= 1:
                if i < self.branches-1:
                    y = torch.cat([features[i], tensors[i], guides[i]], dim=1)
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

        x3d = self.conv3d(x3d)
        x3d = torch.tan(x3d)
        x3d = self.res3d(x3d)
        # x3d = torch.tan(x3d)
        # x3d = self.res3d_2(x3d)
        out = self.last(x3d)

        return out


if __name__ == '__main__':
    x = torch.rand((2, 1, 40, 40)).cuda()
    l = torch.rand((2, 20)).cuda()
    model = OurV2(1, 1).cuda()
    # import time
    # for i in range(10):
    #     y = model(x)
    # t0 = time.time()
    # for i in range(10):
    #     y = model(x)
    # t1 = time.time()
    # for i in range(20):
    #     y = model(x)
    # t2 = time.time()
    y = model(x, l)
    print(y.shape)
