"""Load saved checkpoints and compare models on the SAME test loader; also builds comparison grids."""
import json
import os

import torch

from ..models.improved_unet import build_model
from .metrics import evaluate_model, inference_time


def load_checkpoint(model_name, path, cfg, device):
    m = build_model(model_name, cfg["model"]["base_channels"], cfg["model"]["depth"], cfg["model"]["dilations"])
    ck = torch.load(path, map_location=device)
    m.load_state_dict(ck["model_state_dict"])
    return m.to(device).eval()


def compare(models: dict, loader, device, out_json=None, use_lpips=False):
    """models: {"Plain U-Net": model, "Attention+MS": model, ...}"""
    table = {}
    for name, m in models.items():
        r = evaluate_model(m, loader, device, use_lpips)
        r["sec_per_image_256"] = inference_time(m, device)
        table[name] = r
        print(name, r)
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(table, f, indent=1)
    return table


@torch.no_grad()
def save_comparison_grid(models: dict, loader, device, path, n=4):
    """Rows = samples; columns = input | each model | target."""
    import matplotlib.pyplot as plt
    batch = next(iter(loader))
    x, y = batch["input"][:n].to(device), batch["target"][:n]
    outs = {k: m(x).clamp(0, 1).cpu() for k, m in models.items()}
    cols = ["Input"] + list(models) + ["Target"]
    fig, ax = plt.subplots(len(x), len(cols), figsize=(3.2 * len(cols), 3.2 * len(x)), squeeze=False)
    for r in range(len(x)):
        imgs = [x[r].cpu()] + [outs[k][r] for k in models] + [y[r]]
        for c, im in enumerate(imgs):
            ax[r, c].imshow(im.permute(1, 2, 0)); ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(cols[c])
    plt.tight_layout(); os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=130); plt.close()
