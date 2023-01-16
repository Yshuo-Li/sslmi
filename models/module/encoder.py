import torch.nn as nn

from models.module.copy_layers import make_layer


class Encoder(nn.Module):

    def __init__(self, mid_channels, block, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, num_block in enumerate(num_blocks):
            self.blocks.append(make_layer(
                block,
                num_block,
                mid_channels=mid_channels))
            if i > 0:
                self.downs.append(nn.Sequential(
                    nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                    nn.PReLU()))

    def forward(self, x):
        tensor = self.blocks[0](x)
        results = [tensor]
        for i, down in enumerate(self.downs):
            tensor = down(results[-1])
            results.append(self.blocks[i+1](tensor))
        return results
