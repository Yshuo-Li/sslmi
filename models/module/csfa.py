import torch
import torch.nn as nn
from models.module.resize_features import InterpolateResize


class CSFA(nn.Module):

    def __init__(self, mid_channels, branches, attention):
        super().__init__()
        total_channels = int(mid_channels*branches)
        self.branches = branches
        self.crosser = nn.ModuleList()
        self.bodies = nn.ModuleList()
        for _ in range(branches):
            sub_crosser = nn.ModuleList()
            for _ in range(branches):
                sub_crosser.append(InterpolateResize(mid_channels))
            self.crosser.append(sub_crosser)
            self.bodies.append(nn.Sequential(
                attention(total_channels),
                nn.Conv2d(total_channels, mid_channels, 3, 1, 1),
                nn.PReLU()
            ))

    def forward(self, tensors):
        assert len(tensors) == self.branches
        features = []
        for i, x1 in enumerate(tensors):
            sub_features = []
            for j, x2 in enumerate(tensors):
                if i == j:
                    y = x1
                else:
                    target_size = x1.shape[2:]
                    y = self.crosser[i][j](x2, target_size)
                sub_features.append(y)
            feature = torch.cat(sub_features, dim=1)
            feature = self.bodies[i](feature)
            features.append(feature)
        return features


class MFA(nn.Module):

    def __init__(self, mid_channels, out_channels, branches, attention):
        super().__init__()
        total_channels = int(mid_channels*branches)
        self.branches = branches
        self.crosser = nn.ModuleList()
        for _ in range(branches):
            self.crosser.append(InterpolateResize(mid_channels))
        self.body = nn.Sequential(
            attention(total_channels),
            nn.Conv2d(total_channels, out_channels, 3, 1, 1),
            nn.PReLU()
        )

    def forward(self, tensors, target_size):
        assert len(tensors) == self.branches

        features = []
        for i, x in enumerate(tensors):
            y = self.crosser[i](x, target_size)
            features.append(y)
        feature = torch.cat(features, dim=1)
        feature = self.body(feature)
        return feature


if __name__ == '__main__':
    import torch
    from models.module.attention import CBAM
    model = CSFA(16, 3, CBAM)
    mfa = MFA(16, 3, CBAM)

    x = [
        torch.randn(4, 16, 16, 16),
        torch.randn(4, 16, 32, 32),
        torch.randn(4, 16, 64, 64)
    ]

    y = model(x)
    for i in y:
        print(i.shape)
    z = mfa(y, (64, 64))
    print(z.shape)
