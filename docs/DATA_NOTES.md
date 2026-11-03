# Data notes (Member 1 fills this in)

- Flare7K++ contains real + synthetic flare patterns. Pattern count != number of paired photographs.
- Prefer the OFFICIAL generator/loader (generate_flare.ipynb) over an improvised blend formula; change only paths.
- Verify the official loader's returned keys (e.g. which key is the flare-affected image and which is the clean GT)
  before using `OfficialLoaderAdapter(input_key=..., target_key=...)`.
- If the official generator writes pairs to disk, copy/symlink them to `pairs_root/input/` and `pairs_root/target/` with
  identical filenames; name files `<backgroundId>__<flareId>.png` so the split groups by background.
- Do not assume a mask is flare-only ground truth; inspect docs/overlays first.
- Record: dataset version, license (S-Lab License 1.0), archive filenames, split file path, seed.

## Checklist
- [ ] >=5 pairs visually checked (same scene, flare only in input, dims match, RGB, range)
- [ ] scattering + reflective examples inspected
- [ ] split JSON saved to Drive (`splits/split_seed42.json`)
- [ ] official real test set untouched
