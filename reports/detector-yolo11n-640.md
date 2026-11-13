# Control detector: YOLO11n at 640 (FR-01), second run

The second run of the control detector. It tries the three fixes proposed in the first report ([detector-yolo11n.md](detector-yolo11n.md), section 7): train longer, use a larger input size, and drop the source whose boxes follow a different rule. The task, classes and code are the same as in the first run. Only [settings.yaml](../settings.yaml) changed.

- Raw numbers: [detector-yolo11n-640.json](detector-yolo11n-640.json) for this model, [detector-yolo11n-512-newtest.json](detector-yolo11n-512-newtest.json) for the first model on the same test set, and [dataset-split-640.json](dataset-split-640.json) for the split.
- The weights and photos are not committed, because the repo is public (C-05).

## 1. What changed

| Setting | First run | Second run | Why |
|---|---|---|---|
| `imgsz` | 512 | **640** | 512 found only 21% of the boxes under 0.1% of the image. |
| `epochs` / `patience` | 12 / 5 | **25 / 10** | The val score was still rising at epoch 12. |
| `exclude_folders` | `lgh-appliance` | `lgh-appliance`, **`roboflow-switch-3inct`** | 3inct boxes the whole switch plate, while every other source boxes each key. The first model drew key boxes inside 3inct plates: 35 of 3inct's 48 false positives lay inside a true plate box. |

Everything else is unchanged: YOLO11n pretrained on COCO, batch 16, seed 0, Ultralytics default augmentation, CPU only, and the dataset at commit `910b3d1`.

The three changes were made together, so this run cannot say how much each one helped by itself.

## 2. Datasets and split

The dataset is the same as in the first run except that `roboflow-switch-3inct` (293 images) is left out. It is excluded from train, val and test alike, because its plate boxes would count the key boxes of a model trained on the other sources as wrong.

| Part | Images | Share | `switch` boxes | `toggle` boxes | `power_strip` boxes | Internal / external images |
|---|---:|---:|---:|---:|---:|---|
| train | 1,784 | 67.7% | 1,442 | 1,012 | 1,144 | 111 / 1,673 |
| val | 315 | 12.0% | 248 | 179 | 212 | 14 / 301 |
| test | 535 | 20.3% | 507 | 340 | 351 | 43 / 492 |
| **total** | **2,634** | 100% | 2,197 (40%) | 1,531 (28%) | 1,707 (31%) | 168 / 2,466 |

- **test** is the dataset `eval` split.
- **val** is 15% of `revise`, held out by whole burst (273 bursts).
- Images per source are in [dataset-split-640.json](dataset-split-640.json).
- Team photos are 6% of the data, and all of them show wall switches.

## 3. Training

- 25 epochs took 172 min on the laptop CPU, about 6.9 min per epoch.
- The best weights (`best.pt`) are from epoch 25.
- On val, P is 0.784, R 0.682, mAP50 0.732 and mAP50-95 0.481.

| Epoch | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 17 | 19 | 21 | 23 | 25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val mAP50 | .459 | .311 | .446 | .554 | .650 | .647 | .636 | .672 | .742 | .701 | .728 | .745 | .732 |
| val mAP50-95 | .245 | .151 | .232 | .300 | .397 | .395 | .367 | .422 | .472 | .446 | .472 | .479 | .481 |

The val score levels off from about epoch 17, so going much past 25 epochs is unlikely to help much.

This val set is not the same as the first run's (3inct is out), so the val numbers of the two runs cannot be compared. The test comparison below uses one test set for both models.

## 4. Evaluation

Both models are measured on the **same test set** (535 images, 1,198 boxes). Each one runs at the size it was trained at. A match needs IoU ≥ 0.5. Section 4.1 is the Ultralytics output, which picks its own confidence cut-off (best F1). Sections 4.2 to 4.4 use the app's cut-off, confidence ≥ 0.25.

### 4.1 Standard detection metrics

| Class | Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| **all** | first (512) | 0.815 | 0.729 | 0.765 | 0.481 |
| | **second (640)** | **0.837** | **0.767** | **0.818** | **0.547** |
| switch | first | 0.776 | 0.635 | 0.691 | 0.417 |
| | **second** | **0.862** | **0.742** | **0.800** | **0.501** |
| toggle | first | 0.848 | 0.723 | 0.776 | 0.486 |
| | **second** | 0.832 | **0.743** | **0.805** | **0.546** |
| power_strip | first | 0.822 | 0.829 | 0.829 | 0.539 |
| | **second** | 0.816 | 0.815 | **0.850** | **0.593** |

### 4.2 Finding the control and telling its kind

| | First (512) | Second (640) |
|---|---:|---:|
| Finding, kind-agnostic: precision | 0.773 | **0.804** |
| Finding, kind-agnostic: recall | 0.761 | **0.803** |
| Controls missed | 286 | **236** |
| Extra boxes | 268 | **234** |
| Kind right among found controls | 99.6% | 99.5% |

Confusion of the second model among the found controls (rows are the true kind, columns the predicted kind):

| true \ predicted | switch | toggle | power_strip |
|---|---:|---:|---:|
| switch | 392 | 5 | 0 |
| toggle | 0 | 263 | 0 |
| power_strip | 0 | 0 | 302 |

Class-aware counts of the second model (the kind must also be right):

| Class | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| switch | 392 | 104 | 115 | 0.790 | 0.773 |
| toggle | 263 | 67 | 77 | 0.797 | 0.774 |
| power_strip | 302 | 68 | 49 | 0.816 | 0.860 |

### 4.3 By box size (recall, kind-agnostic)

| Box area | Boxes | First (512) | Second (640) |
|---|---:|---:|---:|
| < 0.1% | 123 | 0.211 | **0.455** |
| 0.1–1% | 566 | 0.781 | **0.807** |
| 1–5% | 366 | 0.866 | 0.869 |
| > 5% | 143 | 0.888 | **0.916** |

### 4.4 By group and source (kind-agnostic, precision / recall)

| Scope | Boxes | First (512) | Second (640) |
|---|---:|---|---|
| external | 1,100 | 0.784 / 0.770 | **0.811 / 0.814** |
| **internal (team photos)** | 98 | 0.657 / 0.663 | **0.728 / 0.684** |
| `ajm` | 21 | 0.769 / 0.952 | 0.955 / 1.000 |
| `lgh` | 33 | 0.818 / 0.818 | 0.929 / 0.788 |
| `hjw` | 25 | 0.500 / 0.480 | 0.565 / 0.520 |
| `chs` | 19 | 0.375 / 0.316 | 0.368 / 0.368 |
| `roboflow-light-switch-dedf6` | 119 | 0.757 / 0.941 | 0.862 / 0.941 |
| `roboflow-switch-bpnbe` | 73 | 0.753 / 0.918 | 0.872 / 0.932 |
| `roboflow-switch-f6lg3` | 206 | 0.752 / 0.529 | 0.761 / 0.680 |
| `roboflow-light-switch-open-image` | 20 | 0.565 / 0.650 | 0.625 / 0.750 |
| `roboflow-toggle-switch-imwkt` | 120 | 0.887 / 0.917 | 0.926 / 0.933 |
| `web-bing-toggle` | 210 | 0.750 / 0.657 | 0.732 / 0.691 |
| `web-bing-powerstrip` | 280 | 0.863 / 0.900 | 0.828 / 0.879 |
| `zenodo-powerstrip` | 61 | 0.552 / 0.607 | 0.758 / 0.770 |
| `wikimedia-commons-powerstrip` | 10 | 1.000 / 0.800 | 1.000 / 0.900 |

### 4.5 Leakage check

1 of 535 test images is near-identical to a train image. The check uses a 16×16 difference hash with up to 16 different bits.

## 5. Analysis

- **All three fixes moved the numbers the right way.** mAP50 rose from 0.765 to 0.818 and mAP50-95 from 0.481 to 0.547. The model now misses 50 fewer controls and draws 34 fewer extra boxes.
- **The wall switch class gained the most** (mAP50 from 0.69 to 0.80). It had the most photos but the lowest score, because of mixed box rules (3inct) and tiny boxes (f6lg3). Fixing the label rule and the input size helped more than adding data would have.
- **Small controls doubled.** Recall on boxes under 0.1% of the image rose from 0.21 to 0.46, and `roboflow-switch-f6lg3` from 0.53 to 0.68. This is the 640 input at work.
- **Telling the kind stays at about 99.5%.** All further work should target finding the control.
- **Power strips barely moved** (mAP50 0.83 to 0.85). `zenodo-powerstrip` improved a lot (recall 0.61 to 0.77), but `web-bing-powerstrip` dropped a little (recall 0.90 to 0.88).
- **Team photos improved less than external photos.** Precision rose from 0.66 to 0.73 and recall from 0.66 to 0.68. `ajm` reaches 1.00 recall, but `chs` stays at 0.37 and `hjw` at 0.52. These look like problems in the photos themselves (angle, distance, light, switch styles), not in the training settings. They need a look at the missed photos and more team photos.

## 6. Limitations

- One run and one seed. The three changes were made together, so their separate effects are not known.
- The test set is the first run's test set without 3inct. It no longer contains photos boxed by plate, so this model has not been checked on that style.
- Team test photos are few (43 images, 98 boxes) and all show wall switches. There is still no number for toggles or power strips in real home photos.
- Power strip photos are mostly clean product photos (Bing), so the power strip score is likely optimistic for camera use.

## 7. Next steps

1. Relabel `roboflow-switch-3inct` per key and bring its 293 images back.
2. Add team photos of all three kinds, especially toggles and power strips taken at home.
3. Look at the missed controls in the `chs` and `hjw` test photos to find what the model struggles with.
4. Do not train beyond about 25 epochs on this data. The val curve is flat from epoch 17.
