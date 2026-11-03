"""Shared, resumable training loop used by BOTH the baseline and the proposed model.

Usage (from a notebook):
    from src.training.train import fit
    fit(cfg, model_name="unet")                       # baseline
    fit(cfg, model_name="attention_multiscale")       # proposed
CLI:
    python -m src.training.train --config configs/default.yaml --model unet
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import yaml
from tqdm import tqdm

from ..data.dataset import build_loaders
from ..models.improved_unet import build_model, count_params
from .losses import build_loss

CKPT_NAMES = {"unet": "baseline/unet", "attention": "improved/attention",
              "multiscale": "improved/multiscale",
              "attention_multiscale": "improved/attention_multiscale"}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def ckpt_paths(drive_root, model_name):
    stem = os.path.join(drive_root, "checkpoints", CKPT_NAMES[model_name])
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    return stem + "_latest.pth", stem + "_best.pth"


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None, amp=False, clip=None):
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, leave=False):
            x, y = batch["input"].to(device), batch["target"].to(device)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                pred = model(x)
            loss = loss_fn(pred.float(), y)   # loss in fp32
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if clip:
                        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    if clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
            if not torch.isfinite(loss):
                raise RuntimeError("NaN/Inf loss: check ranges, lower LR, or disable AMP")
            total += loss.item() * x.size(0); n += x.size(0)
    return total / max(n, 1)


def fit(cfg, model_name=None, resume=True, train_loader=None, val_loader=None):
    model_name = model_name or cfg["model"]["name"]
    t = cfg["train"]
    set_seed(cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if train_loader is None:
        train_loader, val_loader = build_loaders(cfg)

    model = build_model(model_name, cfg["model"]["base_channels"], cfg["model"]["depth"],
                        cfg["model"]["dilations"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=t["amp"] and device.type == "cuda")
    loss_fn = build_loss(cfg)

    root = cfg["project"]["drive_root"]
    latest, best = ckpt_paths(root, model_name)
    log_path = os.path.join(root, "logs", f"{model_name}_log.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    start, best_val, history = 0, float("inf"), []
    if resume and os.path.exists(latest):
        ck = torch.load(latest, map_location=device)
        model.load_state_dict(ck["model_state_dict"]); opt.load_state_dict(ck["optimizer_state_dict"])
        start, best_val, history = ck["epoch"] + 1, ck.get("best_val", float("inf")), ck.get("history", [])
        print(f"Resumed from epoch {start}")

    meta = {"model": model_name, "seed": cfg["project"]["seed"], "crop": cfg["data"]["crop_size"],
            "batch": cfg["data"]["batch_size"], "lr": t["lr"], "loss": t["loss"], "params": count_params(model)}
    print(meta)

    for epoch in range(start, t["epochs"]):
        t0 = time.time()
        tr = run_epoch(model, train_loader, loss_fn, device, opt, scaler, t["amp"], t.get("grad_clip"))
        va = run_epoch(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va, "sec": time.time() - t0})
        print(f"epoch {epoch + 1}/{t['epochs']}  train {tr:.4f}  val {va:.4f}")
        ck = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
              "val_loss": va, "best_val": min(best_val, va), "history": history, "meta": meta}
        torch.save(ck, latest)
        if va < best_val:
            best_val = va; torch.save(ck, best); print("  saved best")
        with open(log_path, "w") as f:
            json.dump({"meta": meta, "history": history}, f, indent=1)
    return model, history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    fit(cfg, a.model, resume=not a.no_resume)
