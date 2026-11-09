"""Convert dataset labels (labeling spec) to a YOLO dataset (FR-01).

- `split: revise` -> train + val (val held out by burst, so near-duplicates stay on one side)
- `split: eval`   -> test, used only by evaluate.py
Images are symlinked, not copied. The output is generated and git-ignored.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from detector import settings


def _bucket(key: str) -> float:
    # burst id를 해시해서 0~1 값으로: 실행할 때마다 같은 분할이 나오도록
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def yolo_lines(record: dict, classes: list[str]) -> list[str]:
    lines = []
    for c in record["controls"]:
        if c.get("decoy"):
            continue
        x, y, w, h = c["bbox"]
        # 이미지 밖으로 나간 부분은 잘라낸다
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(1.0, x + w), min(1.0, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        cls = classes.index(c["kind"])
        lines.append(f"{cls} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
    return lines


def assign(record: dict, val_fraction: float) -> str:
    if record["split"] == "eval":
        return "test"
    if record["split"] != "revise":
        raise ValueError(f"{record['image_id']}: unknown split {record['split']!r}")
    key = record.get("burst") or record["image_id"]
    return "val" if _bucket(key) < val_fraction else "train"


def iter_records(cfg: dict):
    root = settings.resolve(cfg["dataset_root"])
    for src in cfg["sources"]:
        for labels in sorted((root / src).glob("*/labels.jsonl")):
            folder = labels.parent
            if folder.name in cfg["exclude_folders"]:
                continue
            with open(labels) as f:
                for line in f:
                    yield folder, json.loads(line)


def build(cfg: dict) -> dict:
    classes = cfg["classes"]
    out = settings.resolve(cfg["build_dir"])
    for part in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = out / sub / part
            d.mkdir(parents=True, exist_ok=True)
            for old in d.iterdir():
                old.unlink()

    stats = {p: Counter() for p in ("train", "val", "test")}
    train_list = []
    for folder, rec in iter_records(cfg):
        part = assign(rec, cfg["val_fraction"])
        img = folder / "images" / f"{rec['image_id']}.jpg"
        if not img.exists():
            raise FileNotFoundError(img)
        link = out / "images" / part / img.name
        link.symlink_to(img)
        lines = yolo_lines(rec, classes)
        (out / "labels" / part / f"{rec['image_id']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

        kinds = {classes[int(l.split()[0])] for l in lines}
        stats[part]["images"] += 1
        for l in lines:
            stats[part][classes[int(l.split()[0])]] += 1
        if part == "train":
            # 희귀 클래스가 있는 이미지는 목록에 여러 번 넣어 오버샘플링
            times = max([cfg["oversample"].get(k, 1) for k in kinds] or [1])
            train_list += [str(link)] * times

    (out / "train.txt").write_text("\n".join(train_list) + "\n")
    data = {
        "path": str(out),
        "train": "train.txt",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(classes)),
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return {p: dict(c) for p, c in stats.items()} | {"train_list_len": len(train_list)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", type=Path)
    args = ap.parse_args()
    print(json.dumps(build(settings.load(args.settings)["detector"]), indent=2))


if __name__ == "__main__":
    main()
