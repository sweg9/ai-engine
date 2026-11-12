"""Fine-tune YOLO11 on the built dataset (FR-01).

Usage:
    python -m detector.train --name detector-yolo11n

Trains on `train`, picks the weights on `val` (early stopping), and never reads `test`.
Weights go to runs/<name>/weights/ (git-ignored: they are trained on private photos and are
kept out of this public repo, C-05).
"""

import argparse

from ultralytics import YOLO

from detector.settings import ROOT, load, resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="detector-yolo11n")
    ap.add_argument("--epochs", type=int, help="override settings.yaml, for a quick smoke run")
    args = ap.parse_args()
    cfg = load()
    t = cfg["train"]
    model = YOLO(t["model"])
    model.train(
        data=str(resolve(cfg["build_dir"]) / "data.yaml"),
        imgsz=t["imgsz"],
        epochs=args.epochs or t["epochs"],
        patience=t["patience"],
        batch=t["batch"],
        workers=t["workers"],
        device=t["device"],
        seed=cfg["seed"],
        deterministic=True,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
