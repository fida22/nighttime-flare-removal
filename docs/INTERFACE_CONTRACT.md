# Technical interface agreement (agree before integration)

```python
batch = {"input": flare_tensor,   # [B,3,H,W] RGB float in [0,1]
         "target": clean_tensor}  # [B,3,H,W] RGB float in [0,1]
prediction = model(batch["input"])   # [B,3,H,W] RGB, [0,1]
```
- M1 guarantees correct pairing and identical geometric transforms.
- M2/M3 use the same I/O interface, output scaling (sigmoid), split and training settings.
- M4 evaluates every model on the same test images, metric implementation and image range.
- Official real test set stays untouched until final evaluation.
- Split by background (and flare-pattern) identity, not by random crops.
