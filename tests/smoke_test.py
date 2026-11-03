"""Run: python -m tests.smoke_test   (no dataset needed)"""
import torch
from src.models.improved_unet import build_model, count_params
from src.training.losses import L1Loss, L1SSIMLoss
from src.evaluation.metrics import psnr

x = torch.rand(2, 3, 256, 256)
for name in ["unet", "attention", "multiscale", "attention_multiscale"]:
    m = build_model(name, base=16)
    y = m(x)
    assert y.shape == x.shape and 0 <= float(y.min()) and float(y.max()) <= 1, name
    loss = L1SSIMLoss()(y, x); loss.backward()
    print(f"{name:22s} OK  params={count_params(m):,}  loss={loss.item():.3f}")
print("PSNR identical:", psnr(x, x).mean().item() > 90)
print("All smoke tests passed.")
