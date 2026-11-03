"""PSNR / SSIM / optional LPIPS / inference time (Member 4).
All functions take tensors [B,3,H,W] RGB in [0,1]. Same implementation for every model."""
import time

import numpy as np
import torch


def psnr(pred, target, eps=1e-10):
    mse = ((pred.clamp(0, 1) - target) ** 2).flatten(1).mean(1)
    return (10 * torch.log10(1.0 / (mse + eps)))          # [B]


def ssim(pred, target):
    from skimage.metrics import structural_similarity
    out = []
    for p, t in zip(pred.clamp(0, 1).cpu().numpy(), target.cpu().numpy()):
        out.append(structural_similarity(p.transpose(1, 2, 0), t.transpose(1, 2, 0),
                                         channel_axis=2, data_range=1.0))
    return torch.tensor(out)


def get_lpips(device):
    try:
        import lpips
        return lpips.LPIPS(net="alex").to(device).eval()
    except Exception as e:  # optional metric: report only if it works reliably
        print("LPIPS unavailable:", e); return None


@torch.no_grad()
def evaluate_model(model, loader, device, use_lpips=False):
    model.eval().to(device)
    lp = get_lpips(device) if use_lpips else None
    P, S, L = [], [], []
    for b in loader:
        x, y = b["input"].to(device), b["target"].to(device)
        pred = model(x).clamp(0, 1)
        P.append(psnr(pred, y).cpu()); S.append(ssim(pred, y))
        if lp is not None:
            L.append(lp(pred * 2 - 1, y * 2 - 1).flatten().cpu())
    res = {"psnr": torch.cat(P).mean().item(), "ssim": torch.cat(S).mean().item(), "n": int(torch.cat(P).numel())}
    if L:
        res["lpips"] = torch.cat(L).mean().item()
    return res


@torch.no_grad()
def inference_time(model, device, size=(1, 3, 256, 256), warmup=5, runs=20):
    model.eval().to(device)
    x = torch.rand(*size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / runs
