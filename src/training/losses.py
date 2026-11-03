import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(size=11, sigma=1.5, channels=3, device="cpu"):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    w = (g[:, None] @ g[None, :])[None, None]
    return w.expand(channels, 1, size, size).contiguous()


def ssim_torch(x, y, window_size=11, data_range=1.0):
    """Differentiable SSIM, mean over batch. x, y: [B,C,H,W] in [0, data_range]."""
    C = x.shape[1]
    w = _gaussian_window(window_size, 1.5, C, x.device)
    p = window_size // 2
    mu_x, mu_y = F.conv2d(x, w, padding=p, groups=C), F.conv2d(y, w, padding=p, groups=C)
    sxx = F.conv2d(x * x, w, padding=p, groups=C) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=p, groups=C) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=p, groups=C) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.mean()


class L1Loss(nn.Module):
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class L1SSIMLoss(nn.Module):
    """L1 + w * (1 - SSIM). Use only after the plain L1 pipeline works."""
    def __init__(self, ssim_weight=0.1):
        super().__init__(); self.w = ssim_weight

    def forward(self, pred, target):
        return F.l1_loss(pred, target) + self.w * (1 - ssim_torch(pred, target))


def build_loss(cfg):
    t = cfg["train"]
    return L1SSIMLoss(t["ssim_weight"]) if t["loss"] == "l1_ssim" else L1Loss()
