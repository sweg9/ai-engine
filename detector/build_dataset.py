"""Turn the dataset labels into a YOLO dataset (FR-01).

Usage:
    python -m detector.build_dataset

Reads every `labels.jsonl` under `<dataset_root>/internal/*` and `<dataset_root>/external/*`
and writes `<build_dir>/` with images, YOLO label files, `data.yaml` and `stats.json`.
The label files of the dataset are never changed; the class mapping happens here:

- `switch` boxes of the sources in `power_strip_sources` become `power_strip`;
- the earlier kinds are folded with `kind_map` (`push_button` -> `switch`, `knob` and
  `remote` dropped);
- folders in `exclude_folders` are left out.

Split: the dataset's own `split` decides. `eval` -> `test`, never used in training or model
choice. `revise` -> `train`, except about `val_fraction` of it, held out by burst for early
stopping and choosing the weights. A burst (near-identical photos) is always on one side.
"""

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from detector.settings import load, resolve


def to_class(control: dict, source: str, cfg: dict):
    """Return the class name of one labelled control, or None to drop it."""
    kind = cfg["kind_map"].get(control["kind"])
    if kind is None:
        return None
    if kind == "switch" and source in cfg["power_strip_sources"]:
        return "power_strip"
    return kind


def yolo_line(cls_id: int, bbox: list) -> str:
    """bbox is [x, y, w, h], ratios with a top-left origin; YOLO wants the centre."""
    x, y, w, h = bbox
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(1.0, x + w), min(1.0, y + h)
    return f"{cls_id} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def val_bursts(records: list, fraction: float, seed: int) -> set:
    """Pick whole bursts of `revise` records until about `fraction` of the images is held out."""
    sizes = Counter(r["burst"] for r in records)
    order = sorted(sizes, key=lambda b: hashlib.sha1(f"{seed}:{b}".encode()).hexdigest())
    target, held, chosen = fraction * len(records), 0, set()
    for b in order:
        if held >= target:
            break
        chosen.add(b)
        held += sizes[b]
    return chosen


def collect(cfg: dict) -> list:
    root = resolve(cfg["dataset_root"])
    items = []
    for group in ("internal", "external"):
        for labels in sorted((root / group).glob("*/labels.jsonl")):
            source = labels.parent.name
            if source in cfg["exclude_folders"]:
                continue
            files = {json.loads(l)["image_id"]: json.loads(l)["file"]
                     for l in (labels.parent / "manifest.jsonl").read_text(encoding="utf-8").splitlines()}
            for line in labels.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                items.append({
                    "image_id": r["image_id"], "group": group, "source": source, "split": r["split"],
                    "burst": r["burst"], "image": labels.parent / files[r["image_id"]],
                    "boxes": [(to_class(c, source, cfg), c["bbox"]) for c in r["controls"]],
                })
    return items


def main() -> None:
    cfg = load()
    classes = cfg["classes"]
    out = resolve(cfg["build_dir"])
    if out.exists():
        shutil.rmtree(out)
    items = collect(cfg)
    held = val_bursts([i for i in items if i["split"] == "revise"], cfg["val_fraction"], cfg["seed"])
    for i in items:
        i["part"] = "test" if i["split"] == "eval" else ("val" if i["burst"] in held else "train")

    stats = {"parts": defaultdict(lambda: {"images": 0, "boxes": Counter(), "groups": Counter()}),
             "per_source": defaultdict(lambda: Counter()), "dropped_boxes": Counter()}
    index = []
    for i in items:
        part = i["part"]
        (out / "images" / part).mkdir(parents=True, exist_ok=True)
        (out / "labels" / part).mkdir(parents=True, exist_ok=True)
        with Image.open(i["image"]) as im:
            im = im.convert("RGB")
            im.thumbnail((cfg["max_side"], cfg["max_side"]))
            im.save(out / "images" / part / f"{i['image_id']}.jpg", quality=92)
        lines = []
        for cls, bbox in i["boxes"]:
            if cls is None:
                stats["dropped_boxes"][i["source"]] += 1
                continue
            lines.append(yolo_line(classes.index(cls), bbox))
            stats["parts"][part]["boxes"][cls] += 1
        (out / "labels" / part / f"{i['image_id']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["parts"][part]["images"] += 1
        stats["parts"][part]["groups"][i["group"]] += 1
        stats["per_source"][i["source"]][part] += 1
        index.append({"image_id": i["image_id"], "group": i["group"], "source": i["source"], "part": part})

    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names: {json.dumps(dict(enumerate(classes)))}\n")
    (out / "index.json").write_text(json.dumps(index))
    total = len(items)
    summary = {
        "classes": classes,
        "images": total,
        "parts": {p: {"images": s["images"], "share": round(s["images"] / total, 3),
                      "boxes": dict(s["boxes"]), "groups": dict(s["groups"])}
                  for p, s in sorted(stats["parts"].items())},
        "per_source": {k: dict(v) for k, v in sorted(stats["per_source"].items())},
        "dropped_boxes": dict(stats["dropped_boxes"]),
        "val_bursts": len(held),
    }
    (out / "stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("images", "parts", "dropped_boxes")}, indent=2))


if __name__ == "__main__":
    main()
