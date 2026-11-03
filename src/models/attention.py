"""Attention gate and multi-scale dilated bottleneck (Member 3)."""
import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """Additive attention gate (Attention U-Net style).
    g: decoder (gating) feature, x: encoder skip feature; same spatial size here because
    the decoder feature is already upsampled. Returns x weighted by a [B,1,H,W] map in (0,1).
    """
    def __init__(self, c_gate, c_skip, c_inter=None):
        super().__init__()
        c_inter = c_inter or max(c_skip // 2, 1)
        self.wg = nn.Sequential(nn.Conv2d(c_gate, c_inter, 1, bias=False), nn.BatchNorm2d(c_inter))
        self.wx = nn.Sequential(nn.Conv2d(c_skip, c_inter, 1, bias=False), nn.BatchNorm2d(c_inter))
        self.psi = nn.Sequential(nn.Conv2d(c_inter, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        a = self.psi(self.relu(self.wg(g) + self.wx(x)))
        return x * a


class MultiScaleDilatedBottleneck(nn.Module):
    """Parallel 3x3 convs with dilation rates (1,2,4); concat; 1x1 fuse.
    padding == dilation keeps the spatial size unchanged."""
    def __init__(self, cin, cout, dilations=(1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
            for d in dilations])
        self.fuse = nn.Sequential(
            nn.Conv2d(cout * len(dilations), cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.fuse(torch.cat([b(x) for b in self.branches], dim=1))
