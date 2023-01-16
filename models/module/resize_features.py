import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


class Down(nn.Sequential):

    def __init__(self, channels):

        layers = [
            nn.Conv2d(channels, channels*2, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels*2, channels*2, 3, 1, 1),
            nn.LeakyReLU(0.2)]
        super().__init__(*layers)


class Up(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.trans = nn.ConvTranspose2d(
            2*channels, channels, 3, 2, 1, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(2*channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2))

    def forward(self, low, high):

        low = self.trans(low)
        low = F.interpolate(low, high.shape[-2:], mode='nearest')
        high = torch.cat((low, high), dim=1)
        return self.conv(high)


class InterpolateResize(nn.Module):

    def __init__(self,
                 channels,
                 target_size=None,
                 mode='bilinear',
                 conv_flag=True):
        super().__init__()

        self.target_size = target_size
        self.mode = mode
        if conv_flag:
            self.first_conv = nn.Conv2d(channels, channels, 3, 1, 1)
            self.last_conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv_flag = conv_flag

    def forward(self, x, target_size=None):

        if target_size is None:
            target_size = self.target_size
        if self.conv_flag:
            x = self.first_conv(x)
        y = F.interpolate(x, target_size, mode=self.mode, align_corners=False)
        # print(x.shape, y.shape)
        if self.conv_flag:
            y = self.last_conv(y)
        return y
