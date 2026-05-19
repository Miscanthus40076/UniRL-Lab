from __future__ import annotations

from torch import nn

from .encoder import build_mlp
from .transforms import symexp


class MLPDecoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        obs_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = "silu",
        use_symlog: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.use_symlog = bool(use_symlog)
        self.net = build_mlp(feat_dim, self.obs_dim, hidden_dim, num_layers, activation)

    def forward(self, feat):
        pred = self.net(feat)
        return symexp(pred) if self.use_symlog else pred


class CNNDecoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        obs_shape: tuple[int, int, int],
        hidden_dim: int = 256,
        activation: str = "silu",
    ):
        super().__init__()
        channels, height, width = (int(x) for x in obs_shape)
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"CNNDecoder expects height/width divisible by 8, got {obs_shape}")
        act_cls = {"silu": nn.SiLU, "elu": nn.ELU, "relu": nn.ReLU}[activation]
        self.obs_shape = (channels, height, width)
        self.base_h = height // 8
        self.base_w = width // 8
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            act_cls(),
            nn.Linear(hidden_dim, 128 * self.base_h * self.base_w),
            act_cls(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            act_cls(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            act_cls(),
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feat):
        x = self.fc(feat)
        x = x.view(feat.shape[0], 128, self.base_h, self.base_w)
        return self.deconv(x).reshape(feat.shape[0], *self.obs_shape)
