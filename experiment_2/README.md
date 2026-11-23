# Experiment 2 – Compact U-Net and Attention U-Net Experiments

This directory contains four sequential experiments investigating nighttime lens flare and glare removal using compact U-Net architectures.

All experiments were conducted at 256 × 256 resolution using synthetic paired training and validation data.

## Experiment structure

| Directory | Experiment | Description |
|---|---|---|
| experiment_1_compact_unet | Experiment 1 | Compact U-Net baseline trained using ordinary L1 loss. |
| experiment_2_weighted_loss | Experiment 2 | Fine-tuning with a flare-weighted loss to improve flare removal. |
| experiment_3_attention_unet | Experiment 3 | Attention U-Net trained from scratch using a combination of ordinary and flare-weighted L1 losses. |
| experiment_4_attention_finetuning | Experiment 4 | Fine-tuning the Experiment 3 model using square-root residual-weighted L1 and gradient consistency loss. |

Each experiment directory contains:
- `notebooks/`: Training and evaluation notebooks.
- `results/`: Recorded metrics and training history.
- `checkpoints/`: Local model checkpoints. These are excluded from Git tracking.

## Experimental results

### Overall comparison

| Metric | Experiment 1 | Experiment 2 | Experiment 3 | Experiment 4 |
|---|---:|---:|---:|---:|
| Best epoch | 7 | 5 | 20 | 7 |
| Best validation L1 | 0.011117 | 0.010155 | 0.009712 | **0.009193** |
| Synthetic validation PSNR (dB) | 33.88 | **35.23** | 34.42 | 35.14 |
| Synthetic validation SSIM | 0.9680 | 0.9713 | 0.9727 | **0.9750** |
| Input PSNR (dB) | 31.12 | 31.09 | 30.88 | 30.87 |
| Input SSIM | 0.9603 | 0.9609 | 0.9600 | 0.9606 |
| Epochs completed | 10 | 5 | 20 | 9 |

All PSNR and SSIM values above are from synthetic validation data. They are not real-world test results.

### Experiment 1 – Compact U-Net baseline

- Implemented a compact U-Net architecture.
- Trained using ordinary L1 loss.
- Best validation L1: 0.011117 at epoch 7.
- Synthetic validation PSNR: 33.88 dB.
- Synthetic validation SSIM: 0.9680.

### Experiment 2 – Weighted-loss fine-tuning

- Fine-tuned the baseline model using a flare-weighted loss.
- Intended to give greater importance to regions affected by flare.
- Best validation L1: 0.010155 at epoch 5.
- Synthetic validation PSNR: 35.23 dB.
- Synthetic validation SSIM: 0.9713.

### Experiment 3 – Attention U-Net

- Introduced attention gates in a compact U-Net architecture.
- Trained from scratch.
- Architecture: 16–256 channels.
- Trainable parameters: 1,965,223.
- Loss: 0.7 ordinary L1 + 0.3 weighted L1.
- Flare weight: 10.
- Best validation L1: 0.009712 at epoch 20.
- Synthetic validation PSNR: 34.42 dB.
- Synthetic validation SSIM: 0.9727.

Flare-strength analysis:
- Weak flare: L1 increased by approximately 282.28% relative to the input.
- Medium flare: L1 improved by approximately 27.13%.
- Strong flare: L1 improved by approximately 53.90%.

This indicated that the model handled medium and strong flares better but struggled to preserve images with weak flare.

### Experiment 4 – Attention U-Net fine-tuning

- Fine-tuned the best weights from Experiment 3.
- Training budget: 6.5 hours.
- Training data: 10,000 backgrounds.
- Validation data: 1,200 backgrounds.
- Trainable parameters: 1,965,223.
- Learning rate: 0.00002.
- Loss objective:

  `0.65 × ordinary L1 + 0.30 × sqrt-residual-weighted L1 + 0.05 × gradient L1`

- Best validation L1: 0.009193 at epoch 7.
- Synthetic validation PSNR: 35.14 dB.
- Synthetic validation SSIM: 0.9750.

Flare-strength analysis:
- Weak flare: L1 increased by approximately 142.83% relative to the input.
- Medium flare: L1 improved by approximately 29.42%.
- Strong flare: L1 improved by approximately 53.58%.

Fine-tuning improved the overall validation metrics and reduced the weak-flare degradation compared with Experiment 3. However, weak-flare preservation remained a limitation.

## Observations

1. Flare-weighted training improved the synthetic validation metrics over the initial baseline.
2. Attention U-Net achieved a lower validation L1 than the first two experiments.
3. Fine-tuning the attention model achieved the lowest validation L1 and highest synthetic validation SSIM among these four experiments.
4. Flare-strength analysis revealed a consistent limitation: the models improved medium and strong flare images but tended to over-correct weak-flare images.
5. The results are based on synthetic validation data. No real paired test metrics are available in these experiment result files.

## Reproducibility and files

Each notebook contains the corresponding training and evaluation workflow. The JSON files in `results/` record the experiment metrics. Experiments 3 and 4 also include training history files.

Model checkpoints are retained separately and are not committed to GitHub, following the main repository's checkpoint-storage convention. They can be stored in Google Drive for future inference or training resumption.
