| Problem | Likely cause | Action |
|---|---|---|
| CUDA unavailable | no GPU assigned | check runtime; retry later; do CPU tasks |
| CUDA OOM | batch/crop too large | batch 4->2->1; smaller crop; restart runtime |
| Runtime disconnect | free-tier session ended | mount Drive; `fit(...)` resumes from `*_latest.pth` |
| ZIP extraction error | wrong filename / incomplete / disk full | check names and integrity; extract one at a time |
| Input/target mismatch | wrong pairing or transform | stop training; inspect pairs |
| NaN loss | scaling, LR or AMP | check ranges; `train.amp: false`; lower LR |
| Validation >> test | leakage / shift | split by background/pattern id; keep test untouched |
| Light source removed | model confuses source with flare | inspect; consider a verified source-preservation strategy |
| Washed-out output | normalization / range | check RGB order, target scaling, activation, loss |
