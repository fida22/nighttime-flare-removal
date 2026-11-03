# Nighttime Lens Flare Removal (Flare7K++)

Supervised image-to-image restoration: nighttime image with lens flare -> flare-reduced image.
Compares a **plain U-Net baseline** with an **NFR-Net-inspired Attention U-Net with a multi-scale dilated bottleneck**
(dilation 1, 2, 4). The proposed model is a student adaptation, not a reproduction of NFR-Net, and no
performance claim is made until measured on our own untouched test data.

Official dataset / loader: https://github.com/ykdai/Flare7K (S-Lab License 1.0 - review before redistribution).

## Repository layout
```
notebooks/   01_dataset, 02_baseline, 03_improved_model, 04_evaluation, final_pipeline
src/data/        dataset.py, transforms.py          (Member 1)
src/models/      unet.py                            (Member 2)
                 attention.py, improved_unet.py     (Member 3)
src/training/    train.py, losses.py                (shared loop, same settings for all models)
src/evaluation/  metrics.py, evaluate.py            (Member 4)
src/inference/   predict.py                         (Member 4)
configs/default.yaml     shared settings
docs/            team plan, interface contract, data notes, experiment log
results/         figures/, metrics/  (small files only)
```
Large files (archives, extracted data, checkpoints) live in **Google Drive**, never in GitHub.

## Interface contract (do not change silently)
```
batch["input"]  : [B,3,H,W] RGB float32 in [0,1]
batch["target"] : [B,3,H,W] RGB float32 in [0,1]
prediction = model(batch["input"])   # [B,3,H,W] RGB in [0,1]
```

## Quick start (Colab)
1. Mount Drive, `git clone` this repo into `/content`, `pip install -r requirements.txt`.
2. Member 1: download Flare7K++ + background archives (links in the official README) into
   `Drive/Flare_Removal_Project/data/archives/`, extract to `/content/flare_data`, run the OFFICIAL
   generator, write pairs to `/content/flare_pairs/{input,target}/`. Verify >= 5 pairs (notebook 01).
3. Members 2/3 train: `fit(cfg, "unet")` / `fit(cfg, "attention_multiscale")` (notebooks 02/03). Resumable.
4. Member 4 evaluates with `04_evaluation.ipynb`; `final_pipeline.ipynb` loads saved checkpoints only.

Smoke test without data: `python -m tests.smoke_test`

## Ablations
| Exp | `model.name` | Architecture |
|---|---|---|
| A | `unet` | plain U-Net (baseline) |
| B | `attention` | + attention gates |
| C | `multiscale` | + dilated bottleneck |
| D | `attention_multiscale` | both (proposed) |
If time is tight run A and D first.
