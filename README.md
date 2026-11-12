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

### Results at a glance (detector-yolo11n)

Test set: 591 images, 1,254 boxes. Detections need confidence ≥ 0.25 and count as a match at IoU ≥ 0.5.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| **all** | **0.799** | **0.733** | **0.761** | **0.482** |
| switch | 0.743 | 0.650 | 0.686 | 0.423 |
| toggle | 0.833 | 0.718 | 0.770 | 0.484 |
| power_strip | 0.822 | 0.829 | 0.829 | 0.539 |

- **Precision:** of the boxes the model draws, the share that are real controls. Low precision means false alarms.
- **Recall:** of the real controls, the share the model finds. Low recall means missed controls.
- **mAP50:** precision averaged over every confidence cut-off, where a box counts when it overlaps the true box by at least 50% (IoU 0.5). It is one number for "finds the right things in about the right place".
- **mAP50-95:** the same average taken over overlap thresholds from 50% to 95%. It also rewards tight boxes, so it is always lower.

What the numbers mean for FR-01:

| Question | Result |
|---|---|
| Does it find the control, whatever the kind? | precision 0.753, recall 0.767 |
| When it finds one, is the kind right? | **99.3%** (7 wrong out of 958) |
| Team photos against external photos (recall) | 0.66 against 0.78 (only 43 team test images) |
| Small controls, under 0.1% of the image (recall) | 0.21 |
| Large controls, over 1% of the image (recall) | 0.87–0.89 |
| Test images near-identical to a train image | 1 of 591 |

### Data and split

The data is from the private dataset repo at commit `910b3d1`: 2,927 images from 4 team members and 12 external sources. Dataset labels map to the three classes as follows:
- `switch` boxes in a power strip source become `power_strip`.
- `push_button` becomes `switch`.
- `knob` and `remote` are dropped.

| Part | Images | Share | From |
|---|---:|---:|---|
| train | 1,985 | 67.8% | dataset `revise` |
| val | 351 | 12.0% | 15% of `revise`, held out by whole burst (early stopping and picking the weights only) |
| test | 591 | 20.2% | dataset `eval`, never seen in training |

### Method

- **Model:** YOLO11n, COCO-pretrained, fine-tuned for the three classes.
- **Training settings:** image size 512, 12 epochs, batch 16, seed 0, Ultralytics default augmentation.
- **Hardware:** a laptop CPU, which took 81 minutes. The val score was still rising at epoch 12.

### Main findings

- The model is good at telling the kind and weaker at finding the control. More epochs, a larger image size (640) and more team photos are the next steps.
- Many tiny controls are missed. Most come from room photos in `roboflow-switch-f6lg3`. In the app the camera points at the control, so the large-box numbers matter more.
- `roboflow-switch-3inct` boxes the whole plate while other sources box the key. This causes extra boxes, so the team should agree on one box style.

The full write-up has per-source tables, the val curve, the confusion table, limitations and next steps: [reports/detector-yolo11n.md](reports/detector-yolo11n.md).

## Skills and standards

`skills/` is a git submodule of [harness](https://github.com/sweg9/harness). It holds the shared standards and skills for this project. Claude Code finds the skills through `.claude/skills`.

```bash
git clone --recurse-submodules <this repo>
git submodule update --init            # in a repo that is already cloned
git submodule update --remote skills   # move to the latest skills, then commit
```
