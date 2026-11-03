# Team work division (4 members, 14 days)

| Member | Role | Owns | Deliverables |
|---|---|---|---|
| 1 | Dataset & preprocessing | src/data/*, notebooks/01_dataset.ipynb, branch `feature/dataset` | verified pairs (>=5), documented split, test set untouched |
| 2 | Baseline | src/models/unet.py, notebooks/02_baseline.ipynb, `feature/baseline` | baseline checkpoint + logs |
| 3 | Proposed model | src/models/attention.py, improved_unet.py, notebooks/03, `feature/improved-model` | proposed checkpoint + logs + architecture notes |
| 4 | Evaluation & demo | src/evaluation/*, src/inference/*, notebooks/04 + final_pipeline, `feature/evaluation` | metrics table, grids, failure cases, demo |

Dependency: M2/M3 start with dummy tensors (`[2,3,256,256]`) and train on real data only after M1 verifies pairs + split.

## Workflow
clone -> own feature branch -> edit assigned files -> small test -> commit -> push -> PR -> review -> merge.
Never commit secrets, archives, checkpoints; never force-push main; never edit one notebook simultaneously.

## 14-day plan
| Day | M1 Data | M2 Baseline | M3 Proposed | M4 Evaluation |
|---|---|---|---|---|
| 1 | folders; start downloads | baseline file + notebook | improved-model file + notebook | eval notebook + metric plan |
| 2 | extract, inspect | U-Net blocks | attention gate | test metrics on samples |
| 3 | inspect official generator/loader | U-Net forward pass | bottleneck + structure | image-grid helper |
| 4 | generate/verify pairs + split | connect DataLoader; smoke test | dummy-batch test | common tensor format |
| 5 | finalize transforms/loader | baseline pilot | integrate attention + bottleneck | eval script |
| 6 | fix data issues | train baseline | proposed pilot | checkpoint loading check |
| 7 | reproducibility | continue; save ckpt | tune/debug | comparison layout |
| 8 | check split/sample quality | baseline validation | train proposed | check protocol |
| 9 | document dataset | finalize baseline ckpt | continue training | visual comparisons |
| 10 | final data checks | baseline test eval | save best ckpt | run metrics |
| 11 | review failures | share outputs | share outputs | compare PSNR/SSIM + visuals |
| 12 | dataset section | baseline section | architecture section | figures, tables, demo |
| 13 | review reproducibility | verify ckpt | verify ckpt | integrate final notebook |
| 14 | final review | final review | final review | run demo, assemble results |

## Daily routine
Start (5-10 min): today's task + blockers. End (10 min): show one concrete result. Keep `docs/EXPERIMENT_LOG.md` updated.

## Definition of done
1. Official loader works; >=5 pairs checked; ranges/dims correct; split documented; test untouched.
2. U-Net forward works; training runs; checkpoint reloads; val loss + baseline test metrics saved.
3. Attention + multiscale modules work; same interface; checkpoint loading works; architecture documented.
4. Metrics on both models, same test set; comparison images + failure cases saved; final notebook runs.

## Mistakes to avoid
Changing loader output format alone; different normalization/splits for M2/M3; checkpoints only in /content;
big files in GitHub; tuning on the test set; waiting for M1 before writing any model code; integrating in the last hours.
