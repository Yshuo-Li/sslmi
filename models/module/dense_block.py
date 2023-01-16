import torch.nn as nn


class DenseBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels):

        modules = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.LeakyReLU(0.2)]
        super().__init__(*modules)
