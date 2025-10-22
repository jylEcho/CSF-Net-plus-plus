import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


# ---------------------- 基础组件 ----------------------

class ChannelSE(nn.Module):
    """Channel Squeeze-Excitation"""
    def __init__(self, channels, ratio=4):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ---------------------- 编码分支 ----------------------

class Encoder(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        self.base = nn.Conv2d(channels, channels, 3, 1, 1)
        self.mag = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True)
        )
        self.pha = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True)
        )
        self.mag_gate = ChannelSE(channels)
        self.pha_gate = ChannelSE(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        f = torch.fft.rfft2(x, norm='backward')
        mag, pha = torch.abs(f), torch.angle(f)

        mag = self.mag_gate(self.mag(mag))
        pha = self.pha_gate(self.pha(pha))

        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        f_out = torch.complex(real, imag)

        y = torch.fft.irfft2(f_out, s=(H, W), norm='backward')
        return y + self.base(x)


# ---------------------- Norm 层 ----------------------

class LayerNorm2d(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + 1e-5)
        x = x * self.weight.view(1, -1, 1, 1)
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)
        return x


# ---------------------- Transformer 核心 ----------------------

class FeedForward2d(nn.Module):
    def __init__(self, dim, expand=2.0, bias=False):
        super().__init__()
        hidden = int(dim * expand)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=1)
        self.fc2 = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        a, b = self.dw(x).chunk(2, 1)
        return self.fc2(F.gelu(a) * b)


class HybridAttention(nn.Module):
    """Spatial-frequency hybrid attention"""
    def __init__(self, dim, heads=1, bias=False):
        super().__init__()
        self.heads = heads
        self.freq = FourierBlock(dim // 2)
        self.qkv = nn.Conv2d(dim // 2, dim * 3 // 2, 1, bias=bias)
        self.dw = nn.Conv2d(dim * 3 // 2, dim * 3 // 2, 3, 1, 1)
        self.out_proj = nn.Conv2d(dim // 2, dim // 2, 1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        f_out = self.freq(x1)

        b, c, h, w = x2.shape
        qkv = self.dw(self.qkv(x2))
        q, k, v = qkv.chunk(3, 1)

        k, v = self.pool(k), self.pool(v)
        q = rearrange(q, "b (h c) H W -> b h c (H W)", h=self.heads)
        k = rearrange(k, "b (h c) H W -> b h c (H W)", h=self.heads)
        v = rearrange(v, "b (h c) H W -> b h c (H W)", h=self.heads)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (k.transpose(-2, -1) @ q).softmax(dim=-2)
        out = (v @ attn)
        out = rearrange(out, "b h c (H W) -> b (h c) H W", h=self.heads, H=h, W=w)
        return torch.cat([f_out, self.out_proj(out)], dim=1)


class TransformerUnit(nn.Module):
    def __init__(self, dim, heads=1, expand=2.0, bias=False):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = HybridAttention(dim, heads, bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = FeedForward2d(dim, expand, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
