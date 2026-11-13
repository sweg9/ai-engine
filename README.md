# ai-engine

## Control detector (FR-01)

A YOLO11n model that finds controls in camera frames and tells their kind: `switch` (wall switch), `toggle` and `power_strip` (a switch on a power strip), the three kinds of the proposal. Every training and detection setting (model, image size, epochs, split share, confidence and match thresholds, frame stride) is in [settings.yaml](settings.yaml) (C-12).

There are two runs so far. The current settings are those of the second run.

| Run | Report | Settings |
|---|---|---|
| First | [reports/detector-yolo11n.md](reports/detector-yolo11n.md) | 512 px, 12 epochs, every switch source |
| **Second (current)** | [reports/detector-yolo11n-640.md](reports/detector-yolo11n-640.md) | **640 px, 25 epochs, `roboflow-switch-3inct` left out** |

The private [dataset](https://github.com/sweg9/dataset) repo must be checked out next to this one (`../dataset`).

```bash
pip install -r requirements.txt                       # Python 3.11
python -m detector.build_dataset                      # dataset labels -> data/yolo (git-ignored)
python -m detector.train --name detector-yolo11n-640  # weights -> runs/ (git-ignored)
python -m detector.evaluate runs/detector-yolo11n-640/weights/best.pt --out reports/detector-yolo11n-640.json
python -m detector.infer runs/detector-yolo11n-640/weights/best.pt --source 0 --show   # webcam
python -m pytest tests
```

To evaluate weights trained at another size, pass that size, for example `--imgsz 512` for the first run.

This repo is public, so the training photos and the trained weights are never committed: they come from the private dataset (C-05). Rebuild them with the commands above.

### Results at a glance (second run)

Test set: 535 images, 1,198 boxes. A detection counts as a match when it overlaps a true box by IoU ≥ 0.5.

**Standard metrics.** This is the Ultralytics `val` output, the usual way to compare detectors. Ultralytics picks the confidence cut-off that gives the best F1, so precision and recall here are not at the app's 0.25.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| **all** | **0.837** | **0.767** | **0.818** | **0.547** |
| switch | 0.862 | 0.742 | 0.800 | 0.501 |
| toggle | 0.832 | 0.743 | 0.805 | 0.546 |
| power_strip | 0.816 | 0.815 | 0.850 | 0.593 |

**At the app's cut-off (confidence ≥ 0.25).** This is what [detector/infer.py](detector/infer.py) does on a camera. A box counts only when the kind is also right. The lower cut-off keeps more boxes, so recall is higher and precision lower than above.

| Class | Found | Extra | Missed | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| switch | 392 | 104 | 115 | 0.790 | 0.773 |
| toggle | 263 | 67 | 77 | 0.797 | 0.774 |
| power_strip | 302 | 68 | 49 | 0.816 | 0.860 |

mAP50 and mAP50-95 do not depend on the cut-off, so they are the best single numbers to quote.

- **Precision:** of the boxes the model draws, the share that are real controls. Low precision means false alarms.
- **Recall:** of the real controls, the share the model finds. Low recall means missed controls.
- **mAP50:** precision averaged over all recall levels as the confidence cut-off moves from high to low, then averaged over the three classes. A box counts when it overlaps the true box by IoU ≥ 0.5. It is one number for "finds the right things in about the right place".
- **mAP50-95:** the same, averaged over IoU thresholds from 0.50 to 0.95. It also rewards tight boxes, so it is never higher than mAP50.

### Comparison with the first run

Both models are scored on the same test set above, each at the size it was trained at.

| | First (512, 12 epochs) | **Second (640, 25 epochs, no 3inct)** |
|---|---:|---:|
| mAP50 / mAP50-95 | 0.765 / 0.481 | **0.818 / 0.547** |
| mAP50: switch / toggle / power_strip | 0.691 / 0.776 / 0.829 | **0.800 / 0.805 / 0.850** |
| Finding the control, kind-agnostic, confidence ≥ 0.25 (precision / recall) | 0.773 / 0.761 | **0.804 / 0.803** |
| Controls missed / extra boxes | 286 / 268 | **236 / 234** |
| Kind right among found controls | 99.6% | 99.5% |
| Recall on small controls (< 0.1% of the image) | 0.21 | **0.46** |
| Recall on large controls (> 5% of the image) | 0.89 | **0.92** |
| Team photos (precision / recall, 43 test images) | 0.66 / 0.66 | **0.73 / 0.68** |
| Test images near-identical to a train image | 1 of 535 | 1 of 535 |

### Data and split (second run)

The data is from the private dataset repo at commit `910b3d1`: 2,634 images from 4 team members and 10 external sources. Dataset labels map to the three classes as follows:
- `switch` boxes in a power strip source become `power_strip`.
- `push_button` becomes `switch`.
- `knob` and `remote` are dropped.

`roboflow-switch-3inct` (293 images) is left out because it boxes the whole switch plate while every other source boxes each key.

| Part | Images | Share | From |
|---|---:|---:|---|
| train | 1,784 | 67.7% | dataset `revise` |
| val | 315 | 12.0% | 15% of `revise`, held out by whole burst (early stopping and picking the weights only) |
| test | 535 | 20.3% | dataset `eval`, never seen in training |

Train boxes per class: switch 1,442 (40%), toggle 1,012 (28%), power_strip 1,144 (32%). Team photos are 168 of the 2,634 images (6%), all of wall switches.

### Method

- **Model:** YOLO11n, COCO-pretrained, fine-tuned for the three classes.
- **Training settings:** image size 640, 25 epochs (patience 10), batch 16, seed 0, Ultralytics default augmentation.
- **Hardware:** a laptop CPU, which took 172 minutes. The val score levelled off from about epoch 17.

### Main findings

- **Telling the kind is solved; finding the control is what to improve.** Of the controls the model finds, 99.5% get the right kind.
- **The changes helped wall switches the most** (mAP50 0.69 → 0.80). In the first run wall switches had the most photos but the lowest score. Mixed box rules and tiny boxes held them back, and fixing those two raised them the most.
- **Team photos lag behind.** Recall varies widely by person (`ajm` 1.00, `chs` 0.37). There are no team photos of toggles or power strips yet, and power strip photos are mostly clean product photos. The power strip and toggle scores may therefore be optimistic for real camera use.
- **Next steps:**
  - Relabel `roboflow-switch-3inct` per key.
  - Add team photos of toggles and power strips.
  - Look at the photos the model misses.

## Skills and standards

`skills/` is a git submodule of [harness](https://github.com/sweg9/harness). It holds the shared standards and skills for this project. Claude Code finds the skills through `.claude/skills`.

```bash
git clone --recurse-submodules <this repo>
git submodule update --init            # in a repo that is already cloned
git submodule update --remote skills   # move to the latest skills, then commit
```
