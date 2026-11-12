# Control detector: YOLO11n (FR-01)

A first model that finds controls in a camera frame and tells their kind: `switch` (a wall switch), `toggle` (a toggle switch) and `power_strip` (a switch on a power strip). Accuracy was not the goal of this round. The goal was a working pipeline with a recorded dataset, method, split and evaluation, so later models can be compared against it.

- Code: [detector/](../detector). Settings: [settings.yaml](../settings.yaml) (C-12).
- Raw numbers: [detector-yolo11n.json](detector-yolo11n.json) for the evaluation and [dataset-split.json](dataset-split.json) for the split.
- The weights and the training photos are not in this repo, because it is public (C-05). Run the commands in the [README](../README.md) to build them again.

## 1. Datasets

All photos and labels come from the private `dataset` repo at commit `910b3d1` (2026-11-11). The label files there are used as they are. [build_dataset.py](../detector/build_dataset.py) maps them to the three classes when it builds `data/yolo`.

| Group | Source | Images | What it is |
|---|---|---:|---|
| internal | `ajm`, `chs`, `hjw`, `lgh` | 168 | team photos of wall switches |
| external | `roboflow-light-switch-dedf6` | 284 | wall switches |
| external | `roboflow-switch-f6lg3` | 369 | wall switches, often small in a room photo |
| external | `roboflow-switch-3inct` | 293 | wall switches (box covers the whole plate) |
| external | `roboflow-switch-bpnbe` | 185 | wall switches |
| external | `roboflow-light-switch-open-image` | 66 | switches and toggles |
| external | `openimages-light-switch` | 5 | switches and toggles |
| external | `roboflow-toggle-switch-imwkt` | 600 | toggle switches |
| external | `web-bing-toggle` | 404 | toggle switches |
| external | `web-bing-powerstrip` | 455 | power strips with switches |
| external | `zenodo-powerstrip` | 74 | power strips with switches |
| external | `wikimedia-commons-powerstrip` | 24 | power strips with switches |
| | **total** | **2,927** | |

How the dataset labels become the three classes (`settings.yaml` → `detector`):

| Dataset `kind` | Class |
|---|---|
| `switch` in a power strip source (`web-bing-powerstrip`, `zenodo-powerstrip`, `wikimedia-commons-powerstrip`) | `power_strip` |
| `switch` anywhere else | `switch` |
| `toggle` | `toggle` |
| `push_button` (an earlier kind) | `switch` |
| `knob`, `remote` (earlier kinds) | dropped |

The `lgh-appliance` folder is left out because it holds appliance photos, not switches. One box was dropped, from `roboflow-light-switch-open-image`.

## 2. Train, val and test split

| Part | Images | Share | `switch` boxes | `toggle` boxes | `power_strip` boxes | Internal / external images |
|---|---:|---:|---:|---:|---:|---|
| train | 1,985 | 67.8% | 1,630 | 1,028 | 1,144 | 111 / 1,874 |
| val | 351 | 12.0% | 281 | 184 | 212 | 14 / 337 |
| test | 591 | 20.2% | 557 | 346 | 351 | 43 / 548 |
| **total** | **2,927** | 100% | 2,468 | 1,558 | 1,707 | 168 / 2,759 |

- **test** is the dataset's `eval` split, unchanged. The model never sees it during training or when the weights are chosen. Only [evaluate.py](../detector/evaluate.py) reads it.
- **train and val** come from the dataset's `revise` split. About 15% of `revise` (`val_fraction: 0.15`) is held out as val. The holdout is by whole burst (309 bursts), so near-identical photos from one burst never sit on both sides. Bursts are ordered by a hash of their name with `seed: 0`, so every run gives the same split.
- Val is used only for early stopping and for picking `best.pt`.
- Images per source and part are in [dataset-split.json](dataset-split.json) (`per_source`).

## 3. Methodology

| Item | Value |
|---|---|
| Model | Ultralytics YOLO11n (2.6 M parameters), COCO-pretrained `yolo11n.pt` |
| Library | ultralytics 8.4.208, torch 2.13.0 (CPU) |
| Hardware | laptop CPU only (Intel, 8 threads), no GPU |
| Input size | 512 px (`imgsz`). Images are first copied with the long side cut to 1,024 px. |
| Epochs | 12, patience 5, batch 16 |
| Optimiser and augmentation | Ultralytics defaults (`optimizer: auto`, lr0 0.01, flip 0.5, HSV jitter; mosaic only in epochs 1–2 because `close_mosaic: 10` turns it off for the last 10), `deterministic: true`, `seed: 0` |
| Weights kept | `best.pt` by val fitness (here epoch 12) |
| Training time | 81 min (about 7 min per epoch) |

Why YOLO11n: it is the smallest current YOLO model, it trains on a CPU in about an hour, and it runs fast enough for FR-01, which checks every 5th camera frame (`frame_stride: 5`). One model does both finding and typing, so there is no second classifier stage.

Val curve (351 images, 677 boxes):

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mAP50 | .437 | .345 | .402 | .505 | .590 | .617 | .630 | .621 | .667 | .638 | .713 | .714 |
| mAP50-95 | .219 | .205 | .210 | .281 | .322 | .354 | .362 | .348 | .399 | .383 | .427 | .443 |

The curve was still rising at epoch 12, so more epochs should help. They were not run because of the CPU time.

## 4. Evaluation

All numbers are on **test** (591 images, 1,254 boxes) with `best.pt`, `imgsz 512`. The counts below use confidence ≥ 0.25 and a match at IoU ≥ 0.5 (`settings.yaml` → `predict`).

### 4.1 Standard detection metrics (Ultralytics `val`)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| **all** | **0.799** | **0.733** | **0.761** | **0.482** |
| switch | 0.743 | 0.650 | 0.686 | 0.423 |
| toggle | 0.833 | 0.718 | 0.770 | 0.484 |
| power_strip | 0.822 | 0.829 | 0.829 | 0.539 |

### 4.2 Two questions FR-01 asks

1. **Did it find the control?** This is kind-agnostic: a box counts if it overlaps a real control, whatever kind it says. Precision is **0.753** and recall **0.767** (962 found, 292 missed, 316 extra).
2. **When it finds one, is the kind right?** **99.3%** of the found controls get the right kind.

Confusion among the found controls (rows are the true kind, columns the predicted kind):

| true \ predicted | switch | toggle | power_strip |
|---|---:|---:|---:|
| switch | 401 | 4 | 0 |
| toggle | 2 | 258 | 0 |
| power_strip | 1 | 0 | 296 |

Class-aware counts (the kind must also be right):

| Class | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| switch | 402 | 176 | 155 | 0.696 | 0.722 |
| toggle | 258 | 77 | 88 | 0.770 | 0.746 |
| power_strip | 296 | 69 | 55 | 0.811 | 0.843 |

### 4.3 By group and source (kind-agnostic)

| Scope | Boxes | Precision | Recall |
|---|---:|---:|---:|
| external | 1,156 | 0.761 | 0.776 |
| **internal (team photos)** | **98** | **0.657** | **0.663** |
| `ajm` | 21 | 0.769 | 0.952 |
| `lgh` | 33 | 0.818 | 0.818 |
| `hjw` | 25 | 0.500 | 0.480 |
| `chs` | 19 | 0.375 | 0.316 |
| `roboflow-light-switch-dedf6` | 119 | 0.757 | 0.941 |
| `roboflow-switch-bpnbe` | 73 | 0.753 | 0.918 |
| `roboflow-switch-3inct` | 56 | 0.510 | 0.893 |
| `roboflow-switch-f6lg3` | 206 | 0.752 | 0.529 |
| `roboflow-light-switch-open-image` | 20 | 0.565 | 0.650 |
| `roboflow-toggle-switch-imwkt` | 120 | 0.887 | 0.917 |
| `web-bing-toggle` | 210 | 0.750 | 0.657 |
| `web-bing-powerstrip` | 280 | 0.863 | 0.900 |
| `zenodo-powerstrip` | 61 | 0.552 | 0.607 |
| `wikimedia-commons-powerstrip` | 10 | 1.000 | 0.800 |

### 4.4 By box size (share of the image area)

| Box area | Boxes | Found | Recall |
|---|---:|---:|---:|
| < 0.1% | 123 | 26 | 0.211 |
| 0.1–1% | 566 | 442 | 0.781 |
| 1–5% | 370 | 321 | 0.868 |
| > 5% | 195 | 173 | 0.887 |

### 4.5 Leakage check

Each test image was compared with every train image using a 16×16 difference hash, counting a pair as near-identical when the hashes differ in 16 bits or fewer. **1 of 591** test images is near-identical to a train image, so the test numbers are not inflated by copies.

## 5. Analysis

- **Finding the control is the weak part; telling the kind is nearly solved.** 99.3% of found controls get the right kind, but a quarter of controls are missed and a quarter of boxes are extra. Better recall and precision on "is there a control here" matter more than better typing.
- **Small controls are missed.** Boxes under 0.1% of the image are found only 21% of the time. `roboflow-switch-f6lg3` has the most misses of any source (97 of 206 boxes): room photos where the switch is a few pixels wide after resizing to 512 px. A larger `imgsz` (640 or 960) or tiling should help. On a phone camera the user points at the control, so the boxes will mostly be large. For FR-01 the 1–5% and >5% rows matter more.
- **Team photos score lower than external data** (recall 0.66 against 0.78). They are 43 test images, so the number is rough, and the spread between people is large (`ajm` 0.95, `chs` 0.32). The team photos are about 6% of train, so the model mainly learned the look of the external sources. More team photos are the most direct fix, because the app sees photos like these.
- **`roboflow-switch-3inct` has many extra boxes** (precision 0.51). Its boxes cover the whole switch plate while most other sources box the key. The model learned both styles and sometimes outputs both. This is a labelling difference between sources, not just a model error.
- **`zenodo-powerstrip` is hard** (P and R about 0.55–0.61). Its photos are lab photos with many switches in a row and odd angles, unlike the Bing product photos that make up most `power_strip` train data.
- **`roboflow-toggle-switch-imwkt` scores high (0.92 recall) partly because it shows the same few devices** in train and test, although the split is by burst. `web-bing-toggle` (0.66 recall) is the more honest number for unseen toggles.
- **The power strip class works at this stage** (mAP50 0.83). It is defined by the source, not by a separate label, so a switch on a power strip in a source not on the list would be labelled `switch`. If a team photo shows a power strip, it should get its own label (`power_strip`) in the dataset.

## 6. Limitations

- One run, one seed, CPU only, 12 epochs. The val curve had not flattened, so this is not the best this model can do.
- Internal test data is small (43 images, 98 boxes, all wall switches). There are no team photos of toggles or power strips yet, so there is no number for those on real app-like photos.
- The test set is the dataset's `eval` split, which was set before this model. Some sources have few test images (`openimages-light-switch` 1, `wikimedia-commons-powerstrip` 5).
- Label styles differ between sources (plate against key, section 5), which caps the precision any model can reach.
- Kind-agnostic and class-aware matching are run separately, so their counts can differ by one or two (switch: 402 class-aware TP against 401 in the confusion table).
- The `detector-v1` branch (YOLO11s with 5 classes, from another team member) reported kind-agnostic P/R 0.78/0.83. That was a different model size, class set and test data, so it is not a direct comparison. A fair comparison needs both models run with `evaluate.py` on this test split.

## 7. Next steps

1. Train longer (25–50 epochs) and try `imgsz 640`, on a GPU if one is available.
2. Add team photos of all three kinds, especially toggles and power strips.
3. Agree on one box style (key or plate) and relabel `roboflow-switch-3inct` if the team chooses key.
4. Run `detector-v1` through `evaluate.py` on this test split for a fair comparison.
