"""Plain U-Net baseline (Member 2).
Interface: model(x[B,3,H,W] in [0,1]) -> [B,3,H,W] in [0,1] (sigmoid output).
H and W should be divisible by 2**depth (256 crops with depth 4 are fine).
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=32, depth=4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]   # e.g. 32,64,128,256,512
        self.depth = depth
        self.enc = nn.ModuleList()
        c = in_ch
        for ch in chs[:-1]:
            self.enc.append(ConvBlock(c, ch)); c = ch
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self.make_bottleneck(chs[-2], chs[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.up.append(nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2))
            self.dec.append(ConvBlock(chs[i] * 2, chs[i]))
        self.head = nn.Conv2d(chs[0], out_ch, 1)

    def make_bottleneck(self, cin, cout):
        return ConvBlock(cin, cout)

    def skip(self, level, enc_feat, dec_feat):
        """Hook: subclasses (attention) transform the encoder skip before concatenation."""
        return enc_feat

    def forward(self, x):
        skips = []
        for blk in self.enc:
            x = blk(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for j, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = up(x)
            s = skips[self.depth - 1 - j]
            s = self.skip(self.depth - 1 - j, s, x)
            x = dec(torch.cat([s, x], dim=1))
        return torch.sigmoid(self.head(x))
