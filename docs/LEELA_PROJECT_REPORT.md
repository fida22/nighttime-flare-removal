# Nighttime Flare Removal Using Deep Learning

## Project Overview

This project implements a U-Net-based nighttime flare-removal system using the Flare7K++ and Flickr24K datasets.

## Development Environment

- Platform: Kaggle Notebooks
- Framework: PyTorch
- GPU: NVIDIA Tesla T4
- Image resolution: 256 x 256
- Batch size: 8
- Model: U-Net
- Trainable parameters: 7,766,051

## Dataset

- Flickr24K background images: 23,949
- Scattering flare images: 5,000
- Reflective flare images: 2,000
- Training samples: 10,000
- Validation samples: 2,500

## Training

The baseline U-Net was trained for 3 epochs using L1 loss and Adam optimization. FP16 mixed precision and periodic checkpoint saving were used.

The best recorded validation loss was 0.050341.

## Fine-Tuning

Two fine-tuning experiments were conducted:

- Experiment 1: 500 batches, learning rate 0.00001.
- Experiment 2: 500 additional batches, learning rate 0.000001.

## Synthetic Test Results

| Metric | Original Input | Original U-Net | Experiment 1 |
|---|---:|---:|---:|
| MAE | 0.031997 | 0.049669 | 0.045340 |
| PSNR (dB) | 22.8135 | 22.3394 | 22.9760 |

Experiment 1 improved upon the original U-Net, but did not outperform the original input in MAE.

## Ensemble

A three-model prediction ensemble was implemented using the original U-Net, Experiment 1 and Experiment 2. Their predictions are averaged pixel by pixel.

The ensemble's final quantitative evaluation is pending.

## Conclusion

The project demonstrates dataset preparation, U-Net implementation, training, validation, fine-tuning, checkpoint management and image restoration evaluation.

Further evaluation is required to establish whether the ensemble consistently improves flare removal.
