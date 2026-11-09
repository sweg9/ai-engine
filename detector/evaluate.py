"""Measure detector precision and recall per kind on the `eval` 20% (FR-01).

Measuring only: never tune labels, prompts or settings on these numbers.
A detection is a match when kind is equal and IoU >= predict.iou (greedy, highest confidence first).
Label kinds the model was not trained on are mapped through `merge`.
`kind_agnostic` ignores the kind, so models with different classes can be compared.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

from detector import settings
from detector.build_dataset import iter_records


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def match(gts, preds, thr):
    """gts: [(kind, xyxy)], preds: [(kind, conf, xyxy)] -> (tp, fp, fn) per kind."""
    out = defaultdict(lambda: [0, 0, 0])
    used = set()
    for kind, _conf, box in sorted(preds, key=lambda p: -p[1]):
        best, bi = 0.0, None
        for i, (gk, gb) in enumerate(gts):
            if i in used or gk != kind:
                continue
            v = iou(box, gb)
            if v > best:
                best, bi = v, i
        if bi is not None and best >= thr:
            used.add(bi)
            out[kind][0] += 1
        else:
            out[kind][1] += 1
    for i, (gk, _) in enumerate(gts):
        if i not in used:
            out[gk][2] += 1
    return out


def _rel(p: Path) -> str:
    # 리포트에 로컬 절대 경로가 남지 않도록 레포 기준 상대 경로로 기록
    p = p.resolve()
    return str(p.relative_to(settings.REPO)) if p.is_relative_to(settings.REPO) else p.name


def summarize(counts):
    res = {}
    for kind, (tp, fp, fn) in sorted(counts.items()):
        res[kind] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        }
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weights", type=Path)
    ap.add_argument("--settings", type=Path)
    ap.add_argument("--out", type=Path, help="write the JSON report here")
    args = ap.parse_args()
    cfg = settings.load(args.settings)["detector"]
    pcfg, merge = cfg["predict"], cfg.get("merge") or {}
    model = YOLO(str(args.weights))
    classes = [model.names[i] for i in sorted(model.names)]

    groups = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    agnostic = defaultdict(lambda: [0, 0, 0])
    for folder, rec in iter_records(cfg):
        if rec["split"] != "eval":
            continue
        gts = []
        for c in rec["controls"]:
            if c.get("decoy"):
                continue
            x, y, w, h = c["bbox"]
            kind = c["kind"] if c["kind"] in classes else merge.get(c["kind"], c["kind"])
            gts.append((kind, (max(0, x), max(0, y), min(1, x + w), min(1, y + h))))
        r = model.predict(str(folder / "images" / f"{rec['image_id']}.jpg"),
                          conf=pcfg["conf"], device=cfg["train"]["device"], verbose=False)[0]
        preds = [(classes[int(k)], float(s), tuple(b))
                 for k, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxyn.tolist())]
        source = folder.parent.name  # internal / external
        any_v = match([("any", b) for _, b in gts], [("any", s, b) for _, s, b in preds], pcfg["iou"])["any"]
        for key in ("all", source):
            for kind, v in match(gts, preds, pcfg["iou"]).items():
                for i in range(3):
                    groups[key][kind][i] += v[i]
            for i in range(3):
                agnostic[key][i] += any_v[i]

    report = {
        "requirement": "FR-01",
        "weights": _rel(args.weights),
        "split": "eval",
        "conf": pcfg["conf"],
        "iou": pcfg["iou"],
        "classes": classes,
        "by_source": {g: summarize(c) for g, c in groups.items()},
        "kind_agnostic": {g: summarize({"any": v})["any"] for g, v in agnostic.items()},
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
