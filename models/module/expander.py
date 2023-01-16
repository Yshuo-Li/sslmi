import torch.nn as nn
import torch.nn.functional as F

from models.module.copy_layers import make_layer


class Expander(nn.Module):

    def __init__(self, mid_channels):
        super().__init__()
        self.mid_channels = mid_channels

    def forward(self, x, refs):

        # [b, c_in] --> [b, c_out]
        x = x.unsqueeze(1)
        x = F.interpolate(x, self.mid_channels, mode='linear',
                          align_corners=False)
        x = x.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        results = []
        for ref in refs:
            h, w = ref.shape[-2:]
            results.append(x.expand(-1, -1, h, w))

        return results


if __name__ == '__main__':
    from models.module.res_block import ResidualBlockNoBN
    import torch
    x = torch.rand(5, 20)
    model = Expander(64, 4, 40)
    y = model(x)
    for i in y:
        print(i.shape)
