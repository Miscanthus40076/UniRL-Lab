from __future__ import annotations

from torch import nn

from .encoder import build_mlp
from .transforms import symexp, twohot_mean


class ScalarHead(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.net = build_mlp(feat_dim, 1, hidden_dim, num_layers)

    def forward(self, feat):
        return self.net(feat).squeeze(-1)


class ClassHead(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.num_classes = int(num_classes)
        self.net = build_mlp(feat_dim, self.num_classes, hidden_dim, num_layers)

    def forward(self, feat):
        return self.net(feat)


class TwoHotSymlogHead(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_bins: int = 255,
        low: float = -20.0,
        high: float = 20.0,
    ):
        super().__init__()
        self.num_bins = int(num_bins)
        self.low = float(low)
        self.high = float(high)
        self.net = build_mlp(feat_dim, self.num_bins, hidden_dim, num_layers)

    def forward(self, feat):
        return self.net(feat)

    def mean(self, feat):
        logits = self(feat)
        return symexp(twohot_mean(logits, self.num_bins, self.low, self.high))


RewardHead = ScalarHead
ContinueHead = ScalarHead
GraspHead = ScalarHead
