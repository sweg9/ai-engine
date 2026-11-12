# ai-engine

## Control detector (FR-01)

A YOLO11n model that finds controls in camera frames and tells their kind: `switch` (wall switch), `toggle` and `power_strip` (a switch on a power strip), the three kinds of the proposal. The datasets, the method, the train and test split and the evaluation are written up in [reports/detector-yolo11n.md](reports/detector-yolo11n.md). Every number (model, image size, epochs, split share, thresholds, frame stride) is in [settings.yaml](settings.yaml) (C-12).

The private [dataset](https://github.com/sweg9/dataset) repo must be checked out next to this one (`../dataset`).

```bash
pip install -r requirements.txt                       # Python 3.11
python -m detector.build_dataset                      # dataset labels -> data/yolo (git-ignored)
python -m detector.train --name detector-yolo11n      # weights -> runs/ (git-ignored)
python -m detector.evaluate runs/detector-yolo11n/weights/best.pt --out reports/detector-yolo11n.json
python -m detector.infer runs/detector-yolo11n/weights/best.pt --source 0 --show   # webcam
python -m pytest tests
```

This repo is public, so the training photos and the trained weights are never committed: they come from the private dataset (C-05). Rebuild them with the commands above.

## Skills and standards

`skills/` is a git submodule of [harness](https://github.com/sweg9/harness). It holds the shared standards and skills for this project. Claude Code finds the skills through `.claude/skills`.

```bash
git clone --recurse-submodules <this repo>
git submodule update --init            # in a repo that is already cloned
git submodule update --remote skills   # move to the latest skills, then commit
```
