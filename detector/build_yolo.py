"""Build a YOLO detection dataset from the team dataset and the synthetic renders.

Usage (from the repo root):
    python detector/build_yolo.py            # paths and classes come from settings.yaml

Classes are the three control kinds of the team dataset: switch, toggle, power_strip.
Boxes are the moving part (`bbox`), not the plate.

Split: team records keep their own split (revise -> train, eval -> val).
Synthetic scenes go to train, except the last seed folder, which is a synthetic val set
kept apart in val_syn so the real val score is not mixed with it.

Left out:
- roboflow-switch-3inct: its boxes cover the whole plate, not the key.
- records without an image file, and controls without a bbox.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import S, path   # noqa: E402

CLASSES = S["detector"]["classes"]
SKIP_SOURCES = set(S["detector"]["exclude_folders"])
ZOOM = S["detector"]["synthetic_crop_zoom"]


def yolo_line(kind, box):
    x, y, w, h = box
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(1.0, x + w), min(1.0, y + h)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return f"{CLASSES.index(kind)} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def write(out, split, name, lines):
    p = out / "labels" / split / f"{name}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + ("\n" if lines else ""))


def team(dataset, out, stats):
    for labels in sorted(Path(dataset).glob("*/*/labels.jsonl")):
        folder = labels.parent
        if folder.name in SKIP_SOURCES:
            continue
        for line in labels.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            img = next(iter(sorted((folder / "images").glob(r["image_id"] + ".*"))), None)
            if img is None:
                stats["team_missing_image"] += 1
                continue
            split = "val" if r.get("split") == "eval" else "train"
            lines = [l for c in r.get("controls") or [] if c.get("bbox") and c.get("kind") in CLASSES
                     for l in [yolo_line(c["kind"], c["bbox"])] if l]
            name = f"{folder.name}__{r['image_id']}"
            link(img, out / "images" / split / f"{name}{img.suffix.lower()}")
            write(out, split, name, lines)
            stats[f"team_{split}_images"] += 1
            stats[f"team_{split}_boxes"] += len(lines)


def synthetic(root, out, stats, crops_per_scene=1):
    folders = sorted(p for p in Path(root).glob("syn-e*") if (p / "labels.jsonl").exists())
    if not folders:
        return
    val_folder = folders[-1]
    rng = random.Random(0)
    for folder in folders:
        split = "val_syn" if folder == val_folder else "train"
        seen = set()
        for line in (folder / "labels.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["room_id"] in seen:          # one photo per scene: the shots of a scene look alike
                continue
            seen.add(r["room_id"])
            img = folder / "images" / f"{r['image_id']}.jpg"
            if not img.exists():
                continue
            boxes = [c["bbox"] for c in r["controls"] if c.get("bbox")]
            plates = [c["plate_bbox"] for c in r["controls"] if c.get("plate_bbox")]
            name = f"{folder.name}__{r['image_id']}"
            link(img, out / "images" / split / f"{name}.jpg")
            write(out, split, name, [l for b in boxes for l in [yolo_line("switch", b)] if l])
            stats[f"syn_{split}_images"] += 1
            stats[f"syn_{split}_boxes"] += len(boxes)
            # a close crop around the plates, so the keys are not only a few pixels wide
            if split != "train" or not plates:
                continue
            for k in range(crops_per_scene):
                px0 = min(p[0] for p in plates); py0 = min(p[1] for p in plates)
                px1 = max(p[0] + p[2] for p in plates); py1 = max(p[1] + p[3] for p in plates)
                side = max(px1 - px0, (py1 - py0) * 0.75) * rng.uniform(*ZOOM)
                cw, ch = min(1.0, side), min(1.0, side / 0.75)
                cx0 = min(max(0.0, rng.uniform(px1 - cw, px0)), 1 - cw)
                cy0 = min(max(0.0, rng.uniform(py1 - ch, py0)), 1 - ch)
                im = Image.open(img)
                W, H = im.size
                crop = im.crop((int(cx0 * W), int(cy0 * H), int((cx0 + cw) * W), int((cy0 + ch) * H)))
                lines = []
                for x, y, w, h in boxes:
                    nb = [(x - cx0) / cw, (y - cy0) / ch, w / cw, h / ch]
                    if nb[0] >= 0 and nb[1] >= 0 and nb[0] + nb[2] <= 1 and nb[1] + nb[3] <= 1:
                        lines.append(yolo_line("switch", nb))
                cname = f"{name}_crop{k}"
                p = out / "images" / split / f"{cname}.jpg"
                p.parent.mkdir(parents=True, exist_ok=True)
                crop.save(p, quality=92)
                write(out, split, cname, lines)
                stats["syn_train_crops"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(path("dataset_root")))
    ap.add_argument("--synthetic", default=str(path("synthetic_root")))
    ap.add_argument("--out", default=str(path("build_dir")))
    a = ap.parse_args()
    out = Path(a.out)
    for sub in ("images", "labels"):
        for p in (out / sub).glob("*/*"):
            p.unlink()
    stats = Counter()
    team(a.dataset, out, stats)
    synthetic(a.synthetic, out, stats)
    root = out.resolve()
    (out / "data.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    (out / "data_syn.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val_syn\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    print(json.dumps(dict(sorted(stats.items())), indent=1))


if __name__ == "__main__":
    main()
