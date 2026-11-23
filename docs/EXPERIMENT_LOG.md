# Shared experiment log

All runs: `notebooks/Flare_Removal_Final.ipynb` on Kaggle (Tesla T4). Same data cache and schedule for both models in a run.
Test = official Flare7K++ test sets (100 real, 100 synthetic, 512x512). Metrics: PSNR (dB) / SSIM.

| Date | Run | Settings | Real: baseline / proposed | Synthetic: baseline / proposed | Files |
|---|---|---|---|---|---|
| 2026-11-03..16 | template | plain U-Net from scratch, 256 px | output ~= input (no flare removed) | - | `src/` |
| 2026-11-19 | v1 | ResNet-34 U-Net, residual output, 256 px center-crop pairs | 26.09 / 26.23 | 27.41 / 27.53 | `results/earlier_runs/v1_center_crop_256` |
| 2026-11-20 | v2 | pairs at 512 px, random 256 crops | 25.93 / 26.22 | 27.12 / 27.21 | `results/earlier_runs/v2_gen512` |
| 2026-11-20 | v3 | + flare-weighted L1, 70 ep / 80 min | 26.43 / 26.53 | 28.06 / 27.84 | `results/earlier_runs/v3_final_weighted_long` |
| 2026-11-20 | v4 (final) | 120 min per model, 5000 train / 300 val pairs | **26.94 / 26.75** | **28.45 / 28.20** | `results/final_v4` |

## Results table (final run v4, official test sets)
| Model | Real PSNR | Real SSIM | Synth PSNR | Synth SSIM | Params | Time / img |
|---|---|---|---|---|---|---|
| Input (no processing) | 22.56 | 0.8609 | 22.77 | 0.9244 | - | - |
| ResNet-34 U-Net (baseline) | 26.94 | 0.8637 | 28.45 | 0.9376 | 24.5 M | 47 ms |
| + attention + multi-scale (proposed) | 26.75 | 0.8595 | 28.20 | 0.9370 | 26.7 M | 52 ms |

Separate "attention only" / "multi-scale only" ablations were not run, because of the GPU quota.
