# ai-engine

Detector, fingerprints, sound events, VLM prompts and state verification for the hub (FR-01, FR-08, FR-09). The requirements are on the [wiki](https://sweg9.github.io/docs/product/requirements).

## Control detector (FR-01)

A YOLO11 model with the five control kinds of `data-model.md`, in this order: `switch`, `push_button`, `toggle`, `knob`, `remote`. All numbers (model, image size, epochs, held-out share, oversampling, thresholds) are in [settings.yaml](settings.yaml) (C-12).

The [dataset](https://github.com/sweg9/dataset) repo must be checked out next to this one (`../dataset`).

```bash
uv sync                                   # Python 3.11, ultralytics, torch
uv run python -m detector.build_dataset   # labels -> data/yolo (symlinks, git-ignored)
uv run python -m detector.train --name detector-v1
uv run python -m detector.evaluate runs/detector-v1/weights/best.pt --out reports/detector-v1.json
uv run pytest
```

| Dataset split | Used as | Notes |
|---|---|---|
| `revise` | `train` and `val` | `val` is about 15% of `revise`, held out by burst, for early stopping and choosing the weights |
| `eval` | `test` | Only `detector.evaluate` reads it. Never tune labels, settings or prompts on its numbers |

`internal/lgh-appliance/` is left out (appliance photos, not switches). Train images holding a rare kind are listed several times (`oversample`). `merge` folds kinds together for training only, without changing the labels; it is off because the single-class detector-v2 was worse than the five-class detector-v1.

`detector.evaluate` reports precision and recall per kind at `predict.conf` and IoU `predict.iou`, for all of `eval` and separately for `internal` (team photos) and `external` (public datasets). `kind_agnostic` counts a control as found whatever kind was predicted, which is how the hub uses the detector: P-S1 (VLM) decides the kind.

| Model | Classes | Kind-agnostic P / R on `eval` (conf 0.25) | Team photos P / R | Used |
|---|---|---|---|---|
| detector-v1 | 5 | 0.78 / 0.83 | 0.83 / 0.79 | Yes |
| detector-v2 | 1 (all merged) | 0.70 / 0.82 | 0.72 / 0.73 | No |

## Skills and standards

`skills/` is a git submodule of [harness](https://github.com/sweg9/harness). It holds the shared standards and skills for this project. Claude Code finds the skills through `.claude/skills`.

```bash
git clone --recurse-submodules <this repo>
git submodule update --init            # in a repo that is already cloned
git submodule update --remote skills   # move to the latest skills, then commit
```
