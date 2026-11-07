
"""
Shared, resumable training loop for the baseline and proposed models.

Usage from a notebook:

    from src.training.train import fit

    # Baseline
    fit(cfg, model_name="unet",
        train_loader=train_loader,
        val_loader=val_loader)

    # Proposed model
    fit(cfg, model_name="attention_multiscale",
        train_loader=train_loader,
        val_loader=val_loader)

The DataLoaders can be supplied directly from a notebook.
Alternatively, build_loaders(cfg) can be used once
src/data/dataset.py has been implemented.
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

from ..models.improved_unet import build_model, count_params
from .losses import build_loss


# Checkpoint locations for different models
CKPT_NAMES = {
    "unet": "baseline/unet",
    "attention": "improved/attention",
    "multiscale": "improved/multiscale",
    "attention_multiscale": "improved/attention_multiscale"
}


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ckpt_paths(drive_root, model_name):
    """Create checkpoint paths."""
    stem = os.path.join(
        drive_root,
        "checkpoints",
        CKPT_NAMES[model_name]
    )

    os.makedirs(os.path.dirname(stem), exist_ok=True)

    return stem + "_latest.pth", stem + "_best.pth"


def run_epoch(
    model,
    loader,
    loss_fn,
    device,
    optimizer=None,
    scaler=None,
    amp=False,
    clip=None
):
    """Run one training or validation epoch."""

    train = optimizer is not None
    model.train(train)

    total = 0.0
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in tqdm(loader, leave=False):

            x = batch["input"].to(device)
            y = batch["target"].to(device)

            with torch.autocast(
                device_type=device.type,
                enabled=amp and device.type == "cuda"
            ):
                pred = model(x)

            # Calculate loss in float32
            loss = loss_fn(pred.float(), y.float())

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "NaN/Inf loss detected. "
                    "Check image ranges, learning rate, or AMP."
                )

            if train:

                optimizer.zero_grad(set_to_none=True)

                if scaler is not None and scaler.is_enabled():

                    scaler.scale(loss).backward()

                    if clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), clip
                        )

                    scaler.step(optimizer)
                    scaler.update()

                else:

                    loss.backward()

                    if clip:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), clip
                        )

                    optimizer.step()

            total += loss.item() * x.size(0)
            n += x.size(0)

    return total / max(n, 1)


def fit(
    cfg,
    model_name=None,
    resume=True,
    train_loader=None,
    val_loader=None
):
    """
    Train a model using supplied DataLoaders or the configured
    dataset builder.
    """

    model_name = model_name or cfg["model"]["name"]

    if model_name not in CKPT_NAMES:
        raise ValueError(f"Unknown model name: {model_name}")

    t = cfg["train"]

    # Reproducibility
    set_seed(cfg["project"]["seed"])

    # Device selection
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load DataLoaders
    if train_loader is None:

        if val_loader is not None:
            raise ValueError(
                "Provide both train_loader and val_loader, "
                "or neither."
            )

        # Import only when the internal dataset builder is needed
        from ..data.dataset import build_loaders

        train_loader, val_loader = build_loaders(cfg)

    elif val_loader is None:

        raise ValueError(
            "val_loader must be supplied when train_loader is provided."
        )

    # Initialize model
    model = build_model(
        model_name,
        cfg["model"]["base_channels"],
        cfg["model"]["depth"],
        cfg["model"]["dilations"]
    ).to(device)

    # Optimizer
    opt = torch.optim.Adam(
        model.parameters(),
        lr=t["lr"]
    )

    # Mixed precision
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=t["amp"] and device.type == "cuda"
    )

    # Loss function
    loss_fn = build_loss(cfg)

    # Checkpoint and log paths
    root = cfg["project"]["drive_root"]

    latest, best = ckpt_paths(root, model_name)

    log_path = os.path.join(
        root,
        "logs",
        f"{model_name}_log.json"
    )

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Training state
    start = 0
    best_val = float("inf")
    history = []

    # Resume training if a checkpoint exists
    if resume and os.path.exists(latest):

        ck = torch.load(latest, map_location=device)

        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])

        start = ck["epoch"] + 1
        best_val = ck.get("best_val", float("inf"))
        history = ck.get("history", [])

        print(f"Resumed from epoch {start}")

    # Experiment metadata
    meta = {
        "model": model_name,
        "seed": cfg["project"]["seed"],
        "crop": cfg["data"]["crop_size"],
        "batch": cfg["data"]["batch_size"],
        "lr": t["lr"],
        "loss": t["loss"],
        "params": count_params(model)
    }

    print("\nTraining configuration:")
    print(meta)

    # Training loop
    for epoch in range(start, t["epochs"]):

        t0 = time.time()

        # Training
        tr = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer=opt,
            scaler=scaler,
            amp=t["amp"],
            clip=t.get("grad_clip")
        )

        # Validation
        va = run_epoch(
            model,
            val_loader,
            loss_fn,
            device
        )

        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": tr,
            "val_loss": va,
            "sec": elapsed
        })

        print(
            f"Epoch {epoch + 1}/{t['epochs']} "
            f"| Train Loss: {tr:.4f} "
            f"| Val Loss: {va:.4f} "
            f"| Time: {elapsed:.1f}s"
        )

        # Save latest checkpoint
        ck = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "val_loss": va,
            "best_val": min(best_val, va),
            "history": history,
            "meta": meta
        }

        torch.save(ck, latest)

        # Save best checkpoint
        if va < best_val:

            best_val = va

            torch.save(ck, best)

            print("Best checkpoint saved.")

        # Save training history
        with open(log_path, "w") as f:
            json.dump(
                {"meta": meta, "history": history},
                f,
                indent=1
            )

    print("\nTraining completed.")
    print("Latest checkpoint:", latest)
    print("Best checkpoint:", best)
    print("Training log:", log_path)

    return model, history


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="configs/default.yaml"
    )

    ap.add_argument(
        "--model",
        default=None
    )

    ap.add_argument(
        "--no-resume",
        action="store_true"
    )

    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))

    fit(
        cfg,
        a.model,
        resume=not a.no_resume
    )
