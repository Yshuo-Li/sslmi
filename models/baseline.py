import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBN2d(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x) + x


class DownBN2d(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        ]
        super().__init__(*layers)


class Trans(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layer2d = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.LeakyReLU()
        )
        self.layer3d = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 1, 1, 0),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.layer2d(x)
        x = x.unsqueeze(2)
        x = self.layer3d(x)
        return x


class UpBN3d(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 4, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.layer(x)
        # print(x.shape)
        return x


class DecBN3d(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class Baseline(nn.Module):

    def __init__(self, size=(20, 80, 80)):
        super().__init__()

        self.size = size

        self.encoder = nn.Sequential(
            DownBN2d(1, 128), ResBN2d(128, 128),
            DownBN2d(128, 256), ResBN2d(256, 256),
            DownBN2d(256, 512), ResBN2d(512, 512),
            DownBN2d(512, 1024), ResBN2d(1024, 1024),
            DownBN2d(1024, 2048), ResBN2d(2048, 2048),
        )
        self.trans = Trans(2048, 2048)
        self.decoder = nn.Sequential(
            UpBN3d(2048, 1024),
            UpBN3d(1024, 512), DecBN3d(512, 512),
            UpBN3d(512, 256), DecBN3d(256, 256),
            UpBN3d(256, 128), DecBN3d(128, 128),
            UpBN3d(128, 64), DecBN3d(64, 64),
            nn.Conv3d(64, 1, 1, 1, 0), nn.LeakyReLU()
        )

    def forward(self, x, size=None):
        if x.shape[-1] < 64:
            x = F.interpolate(x, size=(64, 64), mode='bilinear',
                              align_corners=False)
        x = self.encoder(x)
        x = self.trans(x)
        x = self.decoder(x)
        if size is None:
            size = self.size
        if x.shape[-3:] != size:
            x = F.interpolate(x, size=size, mode='trilinear',
                              align_corners=False)
        return x


if __name__ == '__main__':
    model = Baseline()
    x = torch.rand(1, 1, 40, 40)
    y = model(x)
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
    print(y.shape)
