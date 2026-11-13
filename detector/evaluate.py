"""Evaluate a trained detector on the held-out `test` part (the dataset's `eval` split) (FR-01).

Usage:
    python -m detector.evaluate runs/detector-yolo11n/weights/best.pt --out reports/detector-yolo11n.json

Reports, on `test` only:
- Ultralytics metrics: precision, recall, mAP50 and mAP50-95, overall and per class.
- Our matching at `predict.conf` and IoU `predict.iou_match`, per class, per group
  (`internal` team photos, `external` public data) and per source:
  * class-aware: a detection counts only if its class is right;
  * kind-agnostic: a detection counts whatever class it has ("was the control found?");
  * type accuracy: among controls found, the share whose class is right ("was the type
    told right?"), with a confusion matrix of true class x predicted class.
- Recall by box size (share of the image area), to show where small controls are missed.
- Leakage check: test images that are near duplicates (dHash) of a train image.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from detector.settings import ROOT, load, resolve

SIZE_BINS = [(0.0, 0.001, "<0.1%"), (0.001, 0.01, "0.1-1%"), (0.01, 0.05, "1-5%"), (0.05, 1.01, ">5%")]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gts, preds, thr, class_aware):
    """Greedy match by confidence. gts: [(cls, box)], preds: [(cls, conf, box)].
    Returns a list of (gt index, pred index) pairs."""
    used, pairs = set(), []
    for pi, (pc, _, pb) in sorted(enumerate(preds), key=lambda t: -t[1][1]):
        best, bi = thr, None
        for gi, (gc, gb) in enumerate(gts):
            if gi in used or (class_aware and gc != pc):
                continue
            v = iou(gb, pb)
            if v >= best:
                best, bi = v, gi
        if bi is not None:
            used.add(bi)
            pairs.append((bi, pi))
    return pairs


def read_labels(path: Path):
    out = []
    for line in path.read_text().split("\n"):
        if line.strip():
            c, x, y, w, h = line.split()
            x, y, w, h = map(float, (x, y, w, h))
            out.append((int(c), (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4)}


def dhash(path, n=16):
    g = Image.open(path).convert("L").resize((n + 1, n), Image.LANCZOS)
    p = list(g.get_flattened_data() if hasattr(g, "get_flattened_data") else g.getdata())
    return sum(1 << i for i, (a, b) in enumerate((p[r * (n + 1) + c], p[r * (n + 1) + c + 1])
                                                  for r in range(n) for c in range(n)) if a > b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, help="image size; default settings.yaml (use the size the weights were trained at)")
    args = ap.parse_args()
    cfg = load()
    classes = cfg["classes"]
    build = resolve(cfg["build_dir"])
    conf, thr = cfg["predict"]["conf"], cfg["predict"]["iou_match"]
    model = YOLO(args.weights)
    imgsz = args.imgsz or cfg["train"]["imgsz"]

    # 1. Ultralytics metrics on test.
    m = model.val(data=str(build / "data.yaml"), split="test", imgsz=imgsz,
                  device=cfg["train"]["device"], plots=False, verbose=False,
                  project=str(ROOT / "runs"), name="eval-tmp", exist_ok=True)
    ultra = {"all": {"precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
                     "mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4)}}
    for k, ci in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(k)
        ultra[classes[int(ci)]] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                                   "mAP50": round(float(ap50), 4), "mAP50-95": round(float(ap), 4)}

    # 2. Our matching, per class, group and source.
    index = {i["image_id"]: i for i in json.loads((build / "index.json").read_text()) if i["part"] == "test"}
    aware = defaultdict(lambda: defaultdict(Counter))      # scope -> class -> tp/fp/fn
    agnostic = defaultdict(Counter)                         # scope -> tp/fp/fn
    confusion = defaultdict(Counter)                        # true class -> predicted class
    by_size = defaultdict(Counter)
    images = sorted((build / "images" / "test").glob("*.jpg"))
    for res in model.predict([str(p) for p in images], conf=conf, imgsz=imgsz,
                             device=cfg["train"]["device"], stream=True, verbose=False):
        path = Path(res.path)
        info = index[path.stem]
        scopes = ["all", info["group"], "source:" + info["source"]]
        gts = read_labels(build / "labels" / "test" / f"{path.stem}.txt")
        preds = [(int(c), float(s), tuple(b)) for c, s, b in
                 zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxyn.tolist())]
        pa = match(gts, preds, thr, class_aware=True)
        for ci, name in enumerate(classes):
            tp = sum(1 for g, _ in pa if gts[g][0] == ci)
            ng, npred = sum(1 for g in gts if g[0] == ci), sum(1 for p in preds if p[0] == ci)
            for s in scopes:
                aware[s][name].update({"tp": tp, "fp": npred - tp, "fn": ng - tp})
        pg = match(gts, preds, thr, class_aware=False)
        for s in scopes:
            agnostic[s].update({"tp": len(pg), "fp": len(preds) - len(pg), "fn": len(gts) - len(pg)})
        found = {g for g, _ in pg}
        for g, p in pg:
            confusion[classes[gts[g][0]]][classes[preds[p][0]]] += 1
        for gi, (gc, gb) in enumerate(gts):
            area = (gb[2] - gb[0]) * (gb[3] - gb[1])
            label = next(lbl for lo, hi, lbl in SIZE_BINS if lo <= area < hi)
            by_size[label]["total"] += 1
            by_size[label]["found"] += gi in found

    def type_acc(scope_conf):
        right = sum(scope_conf[c][c] for c in scope_conf)
        total = sum(sum(v.values()) for v in scope_conf.values())
        return round(right / total, 4) if total else None

    # 3. Leakage check: test images that look like a train image.
    train_hashes = [dhash(p) for p in (build / "images" / "train").glob("*.jpg")]
    test_hashes = [dhash(p) for p in images]
    near = sum(1 for t in test_hashes if any(bin(t ^ h).count("1") <= 16 for h in train_hashes))

    report = {
        "requirement": "FR-01",
        "weights": Path(args.weights).as_posix(),
        "split": "test (dataset split eval)",
        "test_images": len(images),
        "settings": {"conf": conf, "iou_match": thr, "imgsz": imgsz},
        "ultralytics": ultra,
        "class_aware": {s: {c: prf(**v) for c, v in d.items()} for s, d in sorted(aware.items())},
        "kind_agnostic": {s: prf(**v) for s, v in sorted(agnostic.items())},
        "type_accuracy_of_found": type_acc(confusion),
        "confusion_true_x_pred": {t: dict(v) for t, v in confusion.items()},
        "recall_by_box_area": {k: {"total": v["total"], "found": v["found"],
                                   "recall": round(v["found"] / v["total"], 4)}
                               for k, v in ((lbl, by_size[lbl]) for _, _, lbl in SIZE_BINS) if v["total"]},
        "test_images_near_duplicate_of_train": near,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("ultralytics", "type_accuracy_of_found")}, indent=2))
    print("kind-agnostic all:", report["kind_agnostic"]["all"])


if __name__ == "__main__":
    main()
