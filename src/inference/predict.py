"""Single-image inference. Same RGB conversion / [0,1] scaling as training.
    python -m src.inference.predict --model attention_multiscale --ckpt <path.pth> --image in.png --out out.png
Outputs are an ESTIMATED restoration; fully clipped/lost scene detail cannot be recovered."""
import argparse

import numpy as np
import torch
import yaml
from PIL import Image

from ..models.improved_unet import build_model


@torch.no_grad()
def restore_image(model, image: Image.Image, device, multiple=16):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    x = torch.from_numpy(arr.transpose(2, 0, 1))[None].to(device)
    ph, pw = (-h) % multiple, (-w) % multiple           # pad (reflect) so U-Net down/upsampling aligns
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    y = model(x)[:, :, :h, :w].clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray((y * 255).round().astype(np.uint8))


def load_model(name, ckpt, cfg, device):
    m = build_model(name, cfg["model"]["base_channels"], cfg["model"]["depth"], cfg["model"]["dilations"])
    m.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
    return m.to(device).eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="restored.png")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    restore_image(load_model(a.model, a.ckpt, cfg, dev), Image.open(a.image), dev).save(a.out)
    print("saved", a.out)
